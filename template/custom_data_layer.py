import os
import json
import signal
import atexit
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import aiofiles
import aiosqlite
import sqlite3

from chainlit.data.base import BaseDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.element import ElementDict
from chainlit.logger import logger
from chainlit.step import StepDict
from chainlit.types import (
    Feedback,
    FeedbackDict,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

if TYPE_CHECKING:
    from chainlit.element import Element, ElementDict
    from chainlit.step import StepDict

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

class CustomDataLayer(BaseDataLayer):
    def __init__(
        self,
        database_url: str,
    ):
        self._database_url = database_url

    async def get_current_timestamp(self) -> datetime:
        # return datetime.now()
        return datetime.now().isoformat() + "Z"

    async def execute_query(
        self, query: str, params: Union[List, None] = None,
    ) -> List[Dict[str, Any]]:
        try:
            async with aiosqlite.connect(self._database_url) as connection:
                try:
                    cursor = await connection.execute(query, params or [])
                    desc = cursor.description
                    keys = []
                    if desc:
                        keys = [d[0] for d in desc]
                    records = await cursor.fetchall()
                    await cursor.close()
                    await connection.commit()

                    return [{k: v for k, v in zip(keys, record)} for record in records]
                except Exception as e:
                    logger.error(f"Database error: {e!s}")
                    raise
        except aiosqlite.Error as e:
            # Handle connection issues by cleaning up and rethrowing
            logger.error(f"Connection error: {e!s}")
            raise

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        query = """
        SELECT * FROM "User" 
        WHERE identifier = ?
        """
        result = await self.execute_query(query, [identifier])
        if not result or len(result) == 0:
            return None
        row = result[0]
        logger.info(f"user created at {row.get('createdAt')} ({type(row.get('createdAt'))})")

        return PersistedUser(
            id=str(row.get("id")),
            identifier=str(row.get("identifier")),
            createdAt=row.get("createdAt"),
            metadata=json.loads(row.get("metadata", "{}")),
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        query = """
        INSERT INTO "User" (id, identifier, metadata, "createdAt", "updatedAt")
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (identifier) DO UPDATE
        SET metadata = excluded.metadata
        RETURNING *
        """
        now = await self.get_current_timestamp()
        params = [
            str(os.getuid()),
            user.identifier,
            json.dumps(user.metadata),
            now,
            now,
        ]

        result = await self.execute_query(query, params)
        row = result[0]

        return PersistedUser(
            id=str(row.get("id")),
            identifier=str(row.get("identifier")),
            createdAt=row.get("createdAt"),
            metadata=json.loads(row.get("metadata", "{}")),
        )

    async def delete_feedback(self, feedback_id: str) -> bool:
        query = """
        DELETE FROM "Feedback" WHERE id = ?
        """
        await self.execute_query(query, [feedback_id])
        return True

    async def upsert_feedback(self, feedback: Feedback) -> str:
        query = """
        INSERT INTO "Feedback" (id, "forId", name, value, comment)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE
        SET value = excluded.value, comment = excluded.comment
        RETURNING id
        """
        feedback_id = feedback.id or str(os.getuid())
        params = [
            feedback_id,
            feedback.forId,
            "user_feedback",
            float(feedback.value),
            feedback.comment,
        ]
        results = await self.execute_query(query, params)
        return str(results[0]["id"])

    @queue_until_user_message()
    async def create_element(self, element: "Element"):
        if not element.for_id:
            return

        if element.thread_id:
            query = 'SELECT id FROM "Thread" WHERE id = ?'
            results = await self.execute_query(query, [element.thread_id])
            if not results:
                await self.update_thread(thread_id=element.thread_id)

        if element.for_id:
            query = 'SELECT id FROM "Step" WHERE id = ?'
            results = await self.execute_query(query, [element.for_id])
            if not results:
                await self.create_step(
                    {
                        "id": element.for_id,
                        "metadata": {},
                        "type": "run",
                        "start": await self.get_current_timestamp(),
                        "end": await self.get_current_timestamp(),
                    }
                )
        content: Optional[Union[bytes, str]] = None

        if element.path:
            async with aiofiles.open(element.path, "rb") as f:
                content = await f.read()
        elif element.content:
            content = element.content
        elif not element.url:
            raise ValueError("Element url, path or content must be provided")

        if element.thread_id:
            path = f"threads/{element.thread_id}/files/{element.id}"
        else:
            path = f"files/{element.id}"

        query = """
        INSERT INTO "Element" (
            id, "threadId", "forId", metadata, mime, name, "objectKey", url,
            "chainlitKey", display, size, language, page, props
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT (id) DO UPDATE SET
            props = EXCLUDED.props
        """
        params = [
            element.id,
            element.thread_id,
            element.for_id,
            element.type,
            # json.dumps(
            #     {
            #         "size": element.size,
            #         "language": element.language,
            #         "display": element.display,
            #         "type": element.type,
            #         "page": getattr(element, "page", None),
            #     }
            # ),
            element.mime,
            element.name,
            path,
            element.url,
            element.chainlit_key,
            element.display,
            element.size,
            element.language,
            getattr(element, "page", None),
            json.dumps(getattr(element, "props", {})),
        ]
        await self.execute_query(query, params)

    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional[ElementDict]:
        query = """
        SELECT * FROM "Element"
        WHERE id = ? AND "threadId" = ?
        """
        results = await self.execute_query(query, [element_id, thread_id])

        if not results:
            return None

        row = results[0]
        # metadata = json.loads(row.get("metadata", "{}"))

        return ElementDict(
            id=str(row["id"]),
            threadId=str(row["threadId"]),
            type=row.get("type", "file"),
            # type=metadata.get("type", "file"),
            url=str(row["url"]),
            name=str(row["name"]),
            mime=str(row["mime"]),
            objectKey=str(row["objectKey"]),
            forId=str(row["forId"]),
            chainlitKey=row.get("chainlitKey"),
            display=row["display"],
            size=row["size"],
            language=row["language"],
            page=row["page"],
            autoPlay=row.get("autoPlay"),
            playerConfig=row.get("playerConfig"),
            props=json.loads(row.get("props", "{}")),
        )

    @queue_until_user_message()
    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        query = """
        DELETE FROM "Element" 
        WHERE id = ?
        """
        keys = ["id"]
        params = [element_id]

        if thread_id:
            query += ' AND "threadId" = ?'
            params["thread_id"] = thread_id

        await self.execute_query(query, params, keys)

    @queue_until_user_message()
    async def create_step(self, step_dict: StepDict):
        if step_dict.get("threadId"):
            thread_query = 'SELECT id FROM "Thread" WHERE id = ?'
            thread_results = await self.execute_query(thread_query, [step_dict["threadId"]])

            if not thread_results:
                await self.update_thread(thread_id=step_dict["threadId"])

        if step_dict.get("parentId"):
            parent_query = 'SELECT id FROM "Step" WHERE id = ?'
            parent_results = await self.execute_query(parent_query, [step_dict["parentId"]])

            if not parent_results:
                await self.create_step(
                    {
                        "id": step_dict["parentId"],
                        "metadata": {},
                        "type": "run",
                        "createdAt": step_dict.get("createdAt"),
                    }
                )

        query = """
        INSERT INTO "Step" (
            id, "threadId", "parentId", input, metadata, name, output,
            type, "start", "end", "showInput", "isError"
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT (id) DO UPDATE SET
            "parentId" = COALESCE(EXCLUDED."parentId", "Step"."parentId"),
            input = COALESCE(EXCLUDED.input, "Step".input),
            metadata = CASE 
                WHEN EXCLUDED.metadata <> '{}' THEN EXCLUDED.metadata 
                ELSE "Step".metadata 
            END,
            name = COALESCE(EXCLUDED.name, "Step".name),
            output = COALESCE(EXCLUDED.output, "Step".output),
            type = CASE 
                WHEN EXCLUDED.type = 'run' THEN "Step".type 
                ELSE EXCLUDED.type 
            END,
            "threadId" = COALESCE(EXCLUDED."threadId", "Step"."threadId"),
            "end" = COALESCE(EXCLUDED."end", "Step"."end"),
            "start" = MIN(EXCLUDED."start", "Step"."start"),
            "showInput" = COALESCE(EXCLUDED."showInput", "Step"."showInput"),
            "isError" = COALESCE(EXCLUDED."isError", "Step"."isError")
        """

        timestamp = await self.get_current_timestamp()
        created_at = step_dict.get("createdAt")
        # logger.info(f"timestamp: {[timestamp]}")
        if created_at:
            timestamp = created_at
            # timestamp = datetime.strptime(created_at, ISO_FORMAT)
            # logger.info(f"timestamp: {[timestamp]}")

        params = [
            step_dict["id"],
            step_dict.get("threadId"),
            step_dict.get("parentId"),
            step_dict.get("input"),
            json.dumps(step_dict.get("metadata", {})),
            step_dict.get("name"),
            step_dict.get("output"),
            step_dict["type"],
            timestamp,
            timestamp,
            str(step_dict.get("showInput", "json")),
            step_dict.get("isError", False),
        ]

        await self.execute_query(query, params)

    @queue_until_user_message()
    async def update_step(self, step_dict: StepDict):
        await self.create_step(step_dict)

    @queue_until_user_message()
    async def delete_step(self, step_id: str):
        # Delete associated elements and feedbacks first
        await self.execute_query('DELETE FROM "Element" WHERE "forId" = ?', [step_id])
        await self.execute_query('DELETE FROM "Feedback" WHERE "forId" = ?', [step_id])

        # Delete the step
        await self.execute_query('DELETE FROM "Step" WHERE id = ?', [step_id])


    async def get_thread_author(self, thread_id: str) -> str:
        query = """
        SELECT u.identifier 
        FROM "Thread" t
        JOIN "User" u ON t."userId" = u.id
        WHERE t.id = ?
        """
        results = await self.execute_query(query, [thread_id])
        if not results:
            raise ValueError(f"Thread {thread_id} not found")
        return results[0]["identifier"]

    async def delete_thread(self, thread_id: str):
        elements_query = """
        SELECT * FROM "Element" 
        WHERE "threadId" = ?
        """
        elements_results = await self.execute_query(elements_query, [thread_id])

        await self.execute_query('DELETE FROM "Thread" WHERE id = ?', [thread_id])


    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        query = """
        SELECT 
            t.*, 
            u.identifier as user_identifier,
            (SELECT COUNT(*) FROM "Thread" WHERE "userId" = t."userId") as total
        FROM "Thread" t
        LEFT JOIN "User" u ON t."userId" = u.id
        """
        params = []

        if filters.search:
            query += f" AND t.name ILIKE ?"
            params.append(f"%{filters.search}%")

        if filters.userId:
            query += f' AND t."userId" = ?'
            params.append(filters.userId)

        if pagination.cursor:
            query += f' AND t."createdAt" < (SELECT "createdAt" FROM "Thread" WHERE id = ?)'
            params.append(pagination.cursor)

        query += f' ORDER BY t."createdAt" DESC LIMIT ?'
        params.append(pagination.first + 1)

        results = await self.execute_query(query, params)
        threads = results

        has_next_page = len(threads) > pagination.first
        if has_next_page:
            threads = threads[:-1]

        thread_dicts = []
        for thread in threads:
            thread_dict = ThreadDict(
                id=str(thread["id"]),
                createdAt=thread["createdAt"],
                name=thread["name"],
                userId=str(thread["userId"]) if thread["userId"] else None,
                userIdentifier=thread["user_identifier"],
                metadata=json.loads(thread["metadata"]),
                steps=[],
                elements=[],
                tags=[],
            )
            thread_dicts.append(thread_dict)

        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=has_next_page,
                startCursor=thread_dicts[0]["id"] if thread_dicts else None,
                endCursor=thread_dicts[-1]["id"] if thread_dicts else None,
            ),
            data=thread_dicts,
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        query = """
        SELECT t.*, u.identifier as user_identifier
        FROM "Thread" t
        LEFT JOIN "User" u ON t."userId" = u.id
        WHERE t.id = ?
        """
        results = await self.execute_query(query, [thread_id])

        if not results:
            return None

        thread = results[0]

        # Get steps and related feedback
        steps_query = """
        SELECT  s.*, 
                f.id feedback_id, 
                f.value feedback_value, 
                f."comment" feedback_comment
        FROM "Step" s left join "Feedback" f on s.id = f."forId"
        WHERE s."threadId" = ?
        ORDER BY "start"
        """
        steps_results = await self.execute_query(steps_query, [thread_id])

        # Get elements
        elements_query = """
        SELECT * FROM "Element" 
        WHERE "threadId" = ?
        """
        elements_results = await self.execute_query(
            elements_query, [thread_id]
        )

        return ThreadDict(
            id=str(thread["id"]),
            createdAt=thread["createdAt"],
            name=thread["name"],
            userId=str(thread["userId"]) if thread["userId"] else None,
            userIdentifier=thread["user_identifier"],
            metadata=json.loads(thread["metadata"]),
            steps=[self._convert_step_row_to_dict(step) for step in steps_results],
            elements=[
                self._convert_element_row_to_dict(elem) for elem in elements_results
            ],
            tags=[],
        )

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        logger.info(f"aiosqlite: update_thread, thread_id={thread_id}")

        thread_name = truncate(
            name
            if name is not None
            else (metadata.get("name") if metadata and "name" in metadata else None)
        )

        data = {
            "id": thread_id,
            "createdAt": (await self.get_current_timestamp() if metadata is None else None),
            "name": thread_name,
            "userId": user_id,
            "tags": tags,
            "metadata": json.dumps(metadata or {}),
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        # Build the query dynamically based on available fields
        columns = [f'"{k}"' for k in data.keys()]
        placeholders = [f"?" for i in range(len(data))]
        values = list(data.values())

        update_sets = [f'"{k}" = EXCLUDED."{k}"' for k in data.keys() if k != "id"]

        if update_sets:
            query = f"""
                INSERT INTO "Thread" ({", ".join(columns)})
                VALUES ({", ".join(placeholders)})
                ON CONFLICT (id) DO UPDATE
                SET {", ".join(update_sets)};
            """
        else:
            query = f"""
                INSERT INTO "Thread" ({", ".join(columns)})
                VALUES ({", ".join(placeholders)})
                ON CONFLICT (id) DO NOTHING
            """

        await self.execute_query(query, values)

    def _extract_feedback_dict_from_step_row(self, row: Dict) -> Optional[FeedbackDict]:
        if row["feedback_id"] is not None:
            return FeedbackDict(
                forId=row["id"],
                id=row["feedback_id"],
                value=row["feedback_value"],
                comment=row["feedback_comment"],
            )
        return None

    def _convert_step_row_to_dict(self, row: Dict) -> StepDict:
        return StepDict(
            id=str(row["id"]),
            threadId=str(row["threadId"]) if row.get("threadId") else "",
            parentId=str(row["parentId"]) if row.get("parentId") else None,
            name=str(row.get("name")),
            type=row["type"],
            input=row.get("input", {}),
            output=row.get("output", {}),
            metadata=json.loads(row.get("metadata", "{}")),
            createdAt=row["createdAt"] if row.get("createdAt") else None,
            start=row["start"] if row.get("start") else None,
            showInput=row.get("showInput"),
            isError=row.get("isError"),
            end=row["end"] if row.get("end") else None,
            feedback=self._extract_feedback_dict_from_step_row(row),
        )

    def _convert_element_row_to_dict(self, row: Dict) -> ElementDict:
        # metadata = json.loads(row.get("metadata", "{}"))
        return ElementDict(
            id=str(row["id"]),
            threadId=str(row["threadId"]) if row.get("threadId") else None,
            # type=metadata.get("type", "file"),
            type=row.get("type", "file"),
            url=row["url"],
            name=row["name"],
            mime=row["mime"],
            objectKey=row["objectKey"],
            forId=str(row["forId"]),
            chainlitKey=row.get("chainlitKey"),
            display=row["display"],
            size=row["size"],
            language=row["language"],
            page=row["page"],
            # autoPlay=row.get("autoPlay"),
            # playerConfig=row.get("playerConfig"),
            props=json.loads(row.get("props") or "{}"),
        )

    async def build_debug_url(self) -> str:
        return ""

def truncate(text: Optional[str], max_length: int = 255) -> Optional[str]:
    return None if text is None else text[:max_length]
