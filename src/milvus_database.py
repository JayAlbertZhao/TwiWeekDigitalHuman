
# 整体数据存储架构：
# - Milvus 存储原始数据上下文嵌入和单句嵌入，存储对应的SQL索引
# - Milvus 存储完整的摘要数据
# - SQL 按时间和ID存储原始对话

# =========================================================================
# 原始对话数据存储设计 (SQL)
# -------------------------------------------------------------------------
# 表/集合名: user_dialogues
# 描述: 存储每条原始对话记录的完整内容，通过唯一ID和time进行索引。
#
# 字段:
#   id:
#       bigint, 唯一标识每条对话记录，可以与Milvus中的向量ID对应。
#   time:
#       int, Unix时间戳，用于时间排序。
#   role:
#       varchar(100), role id，区分对话中不同对象
#   text:
#       varchar(10k), 原始对话文本。

# 表名： summary
# 描述：存储总结原文等信息
# 字段：
#   id:
#       bigint，与milvus对应
#   start_time:
#       int, 摘要开始的Unix时间戳，作为过滤字段。
#   end_time:
#       int, 摘要结束的Unix时间戳，作为过滤字段。
#   summary_text:
#       varchar(10k), 摘要的原始文本。此字段直接存储在 Milvus 中。

# =========================================================================
# Milvus 向量集合设计 (存储所有向量和部分元数据)
# -------------------------------------------------------------------------

# Collection 1: 原始文本向量集合 (raw_text_embeddings)
# 描述: 存储每条原始对话文本的嵌入向量，用于文本相似度搜索。
#
# 字段:
#   id:
#       bigint, 主键，对应原始对话数据库中的唯一ID。
#   embedding:
#       float vector, 从 'text' 字段生成的嵌入向量。

# ---

# Collection 2: 对话摘要集合 (summary_collection)
# 描述: 直接在 Milvus 中存储摘要的所有信息，包括向量和元数据。
#       该集合中的 id 独立于原始对话 id，专门用于摘要的索引。
#
# 字段:
#   id:
#       bigint, 主键，唯一标识每条摘要记录。
#   embedding:
#       float vector, 从 'summary_text' 字段生成的嵌入向量。

# =========================================================================
# 记忆模块对外接口设计
# -------------------------------------------------------------------------

# 记忆模块为一整个实例，但其中管理若干用户client实例，每个用户client实例管理当前用户
# 每个client对应着四个数据库，即上述四个
# client方法：
# 启动用户接口实例（用户id）
# 功能：启动一个用户接口实例，保存用户信息（目前主要是id）

# 创建用户数据库（（可选）初始化角色的角色名）
# 功能：从指定模板初始化初始记忆库，若未指定则空初始，创建新数据库，包含两个milvus数据库（raw text embedding和summary embedding）和两个SQL数据库（raw text和summary text）

# 插入记录（完整记录的dict，不包含时间，时间由服务端确定）
# 功能：插入记录到SQL库，查询SQL库中时间相邻前面四条记录，五条记录一起嵌入，嵌入和id存入milvus

# 总结记忆（无参数）
# 功能：从总结库获取最大的已总结end_time，将那之后的raw text总结，假定总结的函数已经写好，然后存入总结库

# 查询相关原始记忆（可以接受输入向量或list或文本，但前面这些至少有一个，接受top k默认3）
# 如果向量，直接用向量在raw text里面查询；如果list，假定该list是上下文，嵌入后查询；
# 如果文本，假定是一句话，尝试从SQL里面做完全匹配，如果匹配到，id查询，
# 如果匹配不到，假定是下一句话，用SQL里面最新四句跟这一句嵌入后查询
# 以上几种查询后都rerank，除非k=1

# 查询相关总结记忆（同上）
# 同上

# 关闭用户接口实例（用户id）
# 功能：关闭一个用户接口实例，删除用户信息

import sqlite3
import json
import time
import os
from typing import Optional, Dict, List, Tuple, Union, Any
from collections import deque
import pymilvus
from ABCs import InstantModule

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from pymilvus.model.reranker import BGERerankFunction # 导入BGE Rerank函数
from pymilvus.model.hybrid import BGEM3EmbeddingFunction # 导入BGE M3 Embedding函数

# =========================================================================
# 嵌入函数 (Embedding Function)
# -------------------------------------------------------------------------
class MilvusEmbeddingFunction:
    """
    使用 PyMilvus 提供的 BGEM3EmbeddingFunction 进行向量嵌入。
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", use_fp16: bool = False):
        self.model = BGEM3EmbeddingFunction(model_name=model_name, device=device, use_fp16=use_fp16)
        self.dim = self.model.dim # 获取嵌入维度

    def get_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        生成文本的密集嵌入向量。
        """
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(text)['dense_vecs'].tolist()
        return embeddings

# =========================================================================
# 记忆模块和用户客户端 (MemoryModule & UserClient)
# -------------------------------------------------------------------------

class UserClient:
    """
    管理特定用户的短期记忆、SQL数据库和Milvus向量数据库。
    """
    def __init__(self, user_id: str,
                 embedding_function: MilvusEmbeddingFunction,
                 sql_db_path: Optional[str] = None,
                 milvus_host: str = "localhost", milvus_port: str = "19530"):
        self.user_id = user_id
        self.embedding_function = embedding_function
        self.sql_db_path = sql_db_path if sql_db_path else f"user_data_{user_id}.db"
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        self.sql_conn: Optional[sqlite3.Connection] = None
        self.milvus_alias = f"default_user_{user_id}"

        # 短期记忆库：使用deque实现，限定大小（例如200句或5k tokens）
        self.short_term_memory: deque[str] = deque(maxlen=200) # 暂定200句，后续可根据token量动态调整

        # Milvus Collection 名称
        self.raw_text_collection_name = f"raw_text_embeddings_{user_id}"
        self.summary_collection_name = f"summary_collection_{user_id}"

        # 实例化 Milvus Collection 对象
        self.raw_text_collection: Optional[Collection] = None
        self.summary_collection: Optional[Collection] = None

        # 实例化BGE Rerank函数
        self.bge_reranker = BGERerankFunction(device="cpu") # 假设在CPU上运行，可根据需要修改为cuda

    def _connect_sql(self):
        """连接到SQLite数据库。"""
        if not self.sql_conn:
            # 确保数据库文件所在目录存在
            db_dir = os.path.dirname(self.sql_db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            self.sql_conn = sqlite3.connect(self.sql_db_path)
            self.sql_conn.row_factory = sqlite3.Row # 允许通过名称访问列
        return self.sql_conn

    def _connect_milvus(self):
        """连接到Milvus服务。"""
        if self.milvus_alias not in connections.list_connections():
            connections.connect(alias=self.milvus_alias, host=self.milvus_host, port=self.milvus_port)
        
    def close(self):
        """关闭所有数据库连接。"""
        if self.sql_conn:
            self.sql_conn.close()
            self.sql_conn = None
        try:
            if self.milvus_alias in connections.list_connections():
                connections.remove_connection(self.milvus_alias)
        except Exception as e:
            print(f"Error removing Milvus connection {self.milvus_alias}: {e}")

    def _create_user_databases(self, initial_role: Optional[str] = None):
        """
        创建并初始化用户的SQL和Milvus数据库。
        """
        # 连接SQL并创建表
        conn = self._connect_sql()
        cursor = conn.cursor()

        # 创建 user_dialogues 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_dialogues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL
            );
        """)

        # 创建 summary 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time INTEGER NOT NULL,
                                end_time INTEGER NOT NULL,
                summary_text TEXT NOT NULL
            );
        """)
        conn.commit()

        # 连接Milvus
        self._connect_milvus()
        vector_dim = self.embedding_function.dim # 使用嵌入函数提供的维度

        # 1. 定义 raw_text_embeddings 集合的 Schema
        fields_raw_text = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
        ]
        schema_raw_text = CollectionSchema(fields_raw_text, "存储原始对话文本的嵌入向量")

        # 2. 创建 raw_text_embeddings 集合并为其向量字段创建索引
        if not utility.has_collection(self.raw_text_collection_name, using=self.milvus_alias):
            self.raw_text_collection = Collection(self.raw_text_collection_name, schema_raw_text, using=self.milvus_alias)
            # 为向量字段创建索引
            index_params = {"metric_type":"COSINE", "index_type":"IVF_FLAT", "params":{"nlist":128}}
            self.raw_text_collection.create_index(field_name="embedding", index_params=index_params)
            print(f"Milvus collection '{self.raw_text_collection_name}' created with index.")
        else:
            self.raw_text_collection = Collection(self.raw_text_collection_name, using=self.milvus_alias)
            print(f"Milvus collection '{self.raw_text_collection_name}' already exists.")
        self.raw_text_collection.load()

        # 3. 定义 summary_collection 集合的 Milvus Schema
        fields_summary = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="start_time", dtype=DataType.INT64),
            FieldSchema(name="end_time", dtype=DataType.INT64),
            FieldSchema(name="summary_text", dtype=DataType.VARCHAR, max_length=10000) # 对应SQL中的summary_text
        ]
        schema_summary = CollectionSchema(fields_summary, "存储对话摘要的嵌入向量和元数据")

        # 4. 创建 summary_collection 集合并为其向量字段创建索引
        if not utility.has_collection(self.summary_collection_name, using=self.milvus_alias):
            self.summary_collection = Collection(self.summary_collection_name, schema_summary, using=self.milvus_alias)
            # 为向量字段创建索引
            index_params = {"metric_type":"COSINE", "index_type":"IVF_FLAT", "params":{"nlist":128}}
            self.summary_collection.create_index(field_name="embedding", index_params=index_params)
            print(f"Milvus collection '{self.summary_collection_name}' created with index.")
        else:
            self.summary_collection = Collection(self.summary_collection_name, using=self.milvus_alias)
            print(f"Milvus collection '{self.summary_collection_name}' already exists.")
        self.summary_collection.load()

    def _insert_raw_dialogue_to_sql(self, role: str, text: str) -> int:
        """
        向 `user_dialogues` 表插入原始对话记录。
        Args:
            role (str): 对话角色（例如 "user", "chatbot"）。
            text (str): 原始对话文本。
        Returns:
            int: 插入记录的ID。
        """
        conn = self._connect_sql()
        cursor = conn.cursor()
        current_time = int(time.time())
        cursor.execute("INSERT INTO user_dialogues (time, role, text) VALUES (?, ?, ?);",
                       (current_time, role, text))
        conn.commit()
        return cursor.lastrowid

    def _insert_to_milvus_raw_text(self, record_id: int, embedding: List[float]):
        """
        向 `raw_text_embeddings` 集合插入原始文本的嵌入向量。
        Args:
            record_id (int): 原始对话记录在SQL中的ID。
            embedding (List[float]): 原始对话文本的嵌入向量。
        """
        if not self.raw_text_collection:
            print(f"Error: Milvus collection {self.raw_text_collection_name} not initialized.")
            return
        
        try:
            data = [[record_id], [embedding]]
            self.raw_text_collection.insert(data)
            self.raw_text_collection.flush()
            print(f"Inserted raw text record_id {record_id} to Milvus.")
        except Exception as e:
            print(f"Error inserting raw text record_id {record_id} to Milvus: {e}")

    def _insert_to_milvus_summary(self, record_id: int, start_time: int, end_time: int, summary_text: str):
        """
        向 `summary_collection` 集合插入摘要的嵌入向量和元数据。
        Args:
            record_id (int): 摘要记录的ID。
            start_time (int): 摘要开始时间戳。
            end_time (int): 摘要结束时间戳。
            summary_text (str): 摘要文本。
        """
        if not self.summary_collection:
            print(f"Error: Milvus collection {self.summary_collection_name} not initialized.")
            return
        
        try:
            embedding = self.embedding_function.get_embedding(summary_text)[0]
            data = [
                [record_id],
                [embedding],
                [start_time],
                [end_time],
                [summary_text]
            ]
            self.summary_collection.insert(data)
            self.summary_collection.flush()
            print(f"Inserted summary record_id {record_id} to Milvus.")
        except Exception as e:
            print(f"Error inserting summary record_id {record_id} to Milvus: {e}")

    def _query_milvus_raw_text(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        在 `raw_text_embeddings` 集合中执行向量搜索，并返回匹配的原始对话文本及距离。
        Args:
            query_embedding (List[float]): 查询向量。
            top_k (int): 返回最相似的 top_k 结果。
        Returns:
            List[Dict[str, Any]]: 匹配的原始对话记录列表，包含 `id`, `text`, `distance` 等信息。
        """
        if not self.raw_text_collection:
            print(f"Error: Milvus collection {self.raw_text_collection_name} not initialized.")
            return []

        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.raw_text_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id"]
            )

            retrieved_results = []
            conn = self._connect_sql()
            cursor = conn.cursor()

            for hits in results:
                for hit in hits:
                    # 根据Milvus返回的ID去SQL查询原始文本
                    cursor.execute("SELECT text FROM user_dialogues WHERE id = ?;", (hit.id,))
                    sql_result = cursor.fetchone()
                    if sql_result:
                        retrieved_results.append({"id": hit.id, "text": sql_result["text"], "distance": hit.distance})
            return retrieved_results
        except Exception as e:
            print(f"Error querying Milvus raw text: {e}")
            return []

    def _query_milvus_summary(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        在 `summary_collection` 集合中执行向量搜索，并返回匹配的摘要文本及距离。
        Args:
            query_embedding (List[float]): 查询向量。
            top_k (int): 返回最相似的 top_k 结果。
        Returns:
            List[Dict[str, Any]]: 匹配的摘要记录列表，包含 `id`, `summary_text`, `start_time`, `end_time`, `distance` 等信息。
        """
        if not self.summary_collection:
            print(f"Error: Milvus collection {self.summary_collection_name} not initialized.")
            return []

        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.summary_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id", "start_time", "end_time", "summary_text"]
            )

            retrieved_results = []
            for hits in results:
                for hit in hits:
                    retrieved_results.append({
                        "id": hit.id,
                        "summary_text": hit.entity.get("summary_text"),
                        "start_time": hit.entity.get("start_time"),
                        "end_time": hit.entity.get("end_time"),
                        "distance": hit.distance
                    })
            return retrieved_results
        except Exception as e:
            print(f"Error querying Milvus summary: {e}")
            return []

    def _retrieve_latest_dialogues_from_sql(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        从 `user_dialogues` 表检索最新的对话记录。
        Args:
            count (int): 要检索的记录数量。
        Returns:
            List[Dict[str, Any]]: 最新的对话记录列表，每条记录是一个字典。
        """
        conn = self._connect_sql()
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, time, role, text FROM user_dialogues ORDER BY time DESC LIMIT {count};")
        # SQLite `fetchall` returns a list of sqlite3.Row objects if row_factory is set
        # Convert them to dicts for easier use
        return [dict(row) for row in cursor.fetchall()]

    def _retrieve_summary_from_sql(self, max_end_time: Optional[int] = None, count: int = 1) -> List[Dict[str, Any]]:
        """
        从 `summary` 表检索摘要记录。
        Args:
            max_end_time (Optional[int]): 如果提供，只检索 end_time 小于或等于此值的记录。
            count (int): 要检索的记录数量。
        Returns:
            List[Dict[str, Any]]: 摘要记录列表，每条记录是一个字典。
        """
        conn = self._connect_sql()
        cursor = conn.cursor()
        query = "SELECT id, start_time, end_time, summary_text FROM summary "
        params = []
        if max_end_time is not None:
            query += "WHERE end_time <= ? ORDER BY end_time DESC LIMIT ?;"
            params = [max_end_time, count]
        else:
            query += "ORDER BY end_time DESC LIMIT ?;"
            params = [count]

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def _insert_summary_to_sql(self, start_time: int, end_time: int, summary_text: str) -> int:
        """
        向 `summary` 表插入摘要记录。
        Args:
            start_time (int): 摘要开始的Unix时间戳。
            end_time (int): 摘要结束的Unix时间戳。
            summary_text (str): 摘要的原始文本。
        Returns:
            int: 插入记录的ID。
        """
        conn = self._connect_sql()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO summary (start_time, end_time, summary_text) VALUES (?, ?, ?);",
                       (start_time, end_time, summary_text))
        conn.commit()
        return cursor.lastrowid

    def insert_record(self, record_dict: Dict[str, Any]):
        """
        插入记录到SQL库，查询SQL库中时间相邻前面四条记录，五条记录一起嵌入，嵌入和id存入milvus。
        record_dict 示例: {'role': 'user', 'text': '你好'}
        """
        # 1. 插入记录到SQL库
        role = record_dict.get('role')
        text = record_dict.get('text')
        if not role or not text:
            print("Error: record_dict must contain 'role' and 'text'.")
            return

        record_id = self._insert_raw_dialogue_to_sql(role, text)
        if not record_id:
            print(f"Error: Failed to insert raw dialogue to SQL for text: {text}")
            return

        # 将原始文本添加到短期记忆
        self.short_term_memory.append(text)

        # 2. 查询SQL库中最新五条记录（包括刚刚插入的记录）
        latest_dialogues = self._retrieve_latest_dialogues_from_sql(count=5)
        
        # 3. 拼接对话文本作为上下文
        context_texts = [d['text'] for d in latest_dialogues]
        full_context_text = " ".join(context_texts)

        # 4. 生成嵌入
        if full_context_text:
            context_embedding = self.embedding_function.get_embedding(full_context_text)[0]
            # 5. 将原始记录的ID和生成的嵌入向量插入到Milvus
            self._insert_to_milvus_raw_text(record_id, context_embedding) # 传入embedding
            print(f"Record ID {record_id} and its context embedding inserted to Milvus.")
        else:
            print(f"Warning: No context text to embed for record ID {record_id}.")

    def summarize_memory(self):
        """
        从总结库获取最大的已总结end_time，将那之后的raw text总结，假定总结的函数已经写好，然后存入总结库。
        """
        # 1. 从总结库获取最大的已总结end_time
        latest_summary = self._retrieve_summary_from_sql(count=1)
        max_end_time = latest_summary[0]['end_time'] if latest_summary else 0

        # 2. 检索该end_time之后的所有原始对话记录
        conn = self._connect_sql()
        cursor = conn.cursor()
        cursor.execute("SELECT id, time, role, text FROM user_dialogues WHERE time > ? ORDER BY time ASC;", (max_end_time,))
        raw_dialogues_to_summarize = [dict(row) for row in cursor.fetchall()]

        if not raw_dialogues_to_summarize:
            print("No new raw dialogues to summarize.")
            return

        # 3. 拼接对话文本并进行总结
        texts_to_summarize = [d['text'] for d in raw_dialogues_to_summarize]
        full_text_to_summarize = " ".join(texts_to_summarize)

        # 使用占位符总结函数
        # TODO: 替换为实际的LLM总结函数
        summarized_text = _summarize_placeholder_func(full_text_to_summarize)

        # 获取总结的开始和结束时间
        summary_start_time = raw_dialogues_to_summarize[0]['time']
        summary_end_time = raw_dialogues_to_summarize[-1]['time']

        # 4. 将总结结果插入到SQL summary表
        summary_id = self._insert_summary_to_sql(summary_start_time, summary_end_time, summarized_text)
        if not summary_id:
            print(f"Error: Failed to insert summary to SQL for text: {summarized_text}")
            return

        # 5. 将总结结果插入到Milvus summary_collection
        self._insert_to_milvus_summary(summary_id, summary_start_time, summary_end_time, summarized_text)
        print(f"Summary ID {summary_id} inserted to SQL and Milvus.")

    def query_raw_memory(self, query_data: Union[str, List[str], List[float]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        查询相关原始记忆。
        可以接受输入向量或list或文本，但前面这些至少有一个，接受top k默认3。
        """
        query_embedding: Optional[List[float]] = None
        results: List[Dict[str, Any]] = []
        query_text: Optional[str] = None # 用于Rerank的原始查询文本

        if isinstance(query_data, list) and all(isinstance(i, float) for i in query_data):
            # 如果是向量，直接用向量在raw text里面查询
            query_embedding = query_data
            results = self._query_milvus_raw_text(query_embedding, top_k=top_k)
        elif isinstance(query_data, list) and all(isinstance(i, str) for i in query_data):
            # 如果是list，假定该list是上下文，嵌入后查询
            query_text = " ".join(query_data)
            query_embedding = self.embedding_function.get_embedding(query_text)[0]
            results = self._query_milvus_raw_text(query_embedding, top_k=top_k)
        elif isinstance(query_data, str):
            # 如果是文本，假定是一句话
            query_text = query_data
            conn = self._connect_sql()
            cursor = conn.cursor()

            # 尝试从SQL里面做完全匹配
            cursor.execute("SELECT id, text FROM user_dialogues WHERE text = ?;", (query_data,))
            sql_match = cursor.fetchone()

            if sql_match:
                # 如果匹配到，根据ID查询Milvus
                matched_id = sql_match['id']
                # 生成匹配文本的嵌入，然后进行单次top_k=1的查询来模拟
                query_embedding = self.embedding_function.get_embedding(query_data)[0]
                results = self._query_milvus_raw_text(query_embedding, top_k=top_k) 
                results = [r for r in results if r['id'] == matched_id]
            else:
                # 如果匹配不到，假定是下一句话，用SQL里面最新四句跟这一句嵌入后查询
                latest_dialogues = self._retrieve_latest_dialogues_from_sql(count=4)
                context_texts = [d['text'] for d in latest_dialogues]
                context_texts.append(query_data) # 将新查询文本也加入上下文
                full_context_text = " ".join(context_texts)
                query_embedding = self.embedding_function.get_embedding(full_context_text)[0]
                results = self._query_milvus_raw_text(query_embedding, top_k=top_k)
        else:
            print("Error: Unsupported query_data type.")
            return []

        # Rerank，除非k=1 并且有原始query_text
        if query_text and top_k > 1 and results:
            documents_for_rerank = [r['text'] for r in results]
            reranked_results = self.bge_reranker.rerank(query_text, documents_for_rerank, top_k=top_k)
            
            # 根据rerank结果重新构建原始results列表
            final_results = []
            for r_result in reranked_results:
                original_index = r_result.index
                original_doc = results[original_index]
                original_doc['distance'] = 1 - r_result.score # 将score转换为距离度量
                final_results.append(original_doc)
            return final_results
        
        return results

    def query_summary_memory(self, query_data: Union[str, List[str], List[float]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        查询相关总结记忆。
        """
        query_embedding: Optional[List[float]] = None
        results: List[Dict[str, Any]] = []
        query_text: Optional[str] = None # 用于Rerank的原始查询文本

        if isinstance(query_data, list) and all(isinstance(i, float) for i in query_data):
            # 如果是向量，直接用向量在summary collection里面查询
            query_embedding = query_data
            results = self._query_milvus_summary(query_embedding, top_k=top_k)
        elif isinstance(query_data, list) and all(isinstance(i, str) for i in query_data):
            # 如果是list，假定该list是上下文，嵌入后查询
            query_text = " ".join(query_data)
            query_embedding = self.embedding_function.get_embedding(query_text)[0]
            results = self._query_milvus_summary(query_embedding, top_k=top_k)
        elif isinstance(query_data, str):
            # 如果是文本，假定是一句话，嵌入后查询
            query_text = query_data
            query_embedding = self.embedding_function.get_embedding(query_text)[0]
            results = self._query_milvus_summary(query_embedding, top_k=top_k)
        else:
            print("Error: Unsupported query_data type.")
            return []
        
        # Rerank，除非k=1 并且有原始query_text
        if query_text and top_k > 1 and results:
            documents_for_rerank = [r['summary_text'] for r in results]
            reranked_results = self.bge_reranker.rerank(query_text, documents_for_rerank, top_k=top_k)

            # 根据rerank结果重新构建原始results列表
            final_results = []
            for r_result in reranked_results:
                original_index = r_result.index
                original_doc = results[original_index]
                original_doc['distance'] = 1 - r_result.score # 将score转换为距离度量
                final_results.append(original_doc)
            return final_results

        return results

class MemoryModule(InstantModule):
    """
    记忆模块的整体实例，管理所有用户的UserClient实例。
    """
    def __init__(self, milvus_host: str = "localhost", milvus_port: str = "19530"):
        super().__init__() # 调用父类的__init__方法
        self.user_clients: Dict[str, UserClient] = {}
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

    async def _setup(self):
        # 建立全局Milvus连接，用于管理连接和utility操作
        if not self._is_ready.is_set():
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            self._is_ready.set()

    def start_user_client_instance(self, user_id: str) -> UserClient:
        """
        启动一个用户接口实例，保存用户信息。
        """
        if user_id in self.user_clients:
            print(f"UserClient for {user_id} already exists. Returning existing instance.")
            return self.user_clients[user_id]
        
        # 在这里实例化 MilvusEmbeddingFunction，并传递给 UserClient
        embedding_function_instance = MilvusEmbeddingFunction(
            model_name='BAAI/bge-m3',
            device='cpu',
            use_fp16=False
        )
        
        client = UserClient(user_id=user_id,
                            embedding_function=embedding_function_instance,
                            milvus_host=self.milvus_host,
                            milvus_port=self.milvus_port)
        self.user_clients[user_id] = client
        
        # 自动创建数据库
        client._create_user_databases()
        
        return client

    async def shutdown(self):
        """
        关闭记忆模块，包括所有用户客户端和全局Milvus连接。
        """
        # 关闭所有活跃的用户客户端
        for user_id in list(self.user_clients.keys()):
            self.close_user_client_instance(user_id)
        
        # 移除全局Milvus连接
        try:
            if "default" in connections.list_connections():
                connections.remove_connection("default")
                print("Global Milvus connection 'default' removed.")
        except Exception as e:
            print(f"Error removing global Milvus connection: {e}")

    def close_user_client_instance(self, user_id: str):
        """
        关闭一个用户接口实例，删除用户信息。
        """
        if user_id in self.user_clients:
            self.user_clients[user_id].close()
            del self.user_clients[user_id]
            print(f"UserClient for {user_id} closed and removed.")
        else:
            print(f"UserClient for {user_id} not found.")

    def get_user_client(self, user_id: str) -> Optional[UserClient]:
        """获取指定user_id的UserClient实例。"""
        return self.user_clients.get(user_id)

# 总结函数占位符
def _summarize_placeholder_func(text: str) -> str:
    """
    一个总结函数的占位符。
    实际应用中会是一个调用LLM进行总结的函数。
    """
    return f"Summary of: {text[:50]}..." # 简单截断作为示例
