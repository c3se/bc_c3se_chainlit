import os
import pwd
from typing import Dict, Optional

from openai import AsyncOpenAI
import chainlit as cl

client = AsyncOpenAI(base_url=os.getenv("VLLM_BASE_URL"), api_key=os.getenv("VLLM_API_KEY"))

cl.instrument_openai()

settings = {
    "model": os.getenv("HF_MODEL"),
    "temperature": 0.3,
    "stream": True,
}

mcp_tools_cache = {}

@cl.on_chat_start
async def start():
    username = os.getenv("USER")
    fullname = pwd.getpwnam(username).pw_gecos.split(',')[0]

    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant running locally via vLLM. You can access tools using MCP servers.",
            }
        ],
    )

    await cl.Message(
        content=f"Hi {fullname}! I am your AI assistant on {os.getenv('HOSTNAME')}\n"
    ).send()

@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
    # Verify the signature of a token in the header (ex: jwt token)
    # or check that the value is matching a row from your database
    user = os.getenv("USER")
    cookies = [t.strip() for t in headers.get('cookie').split(";")]

    if f'_chainlit_token={os.getenv("CHAINLIT_AUTH_SECRET")}' in cookies:
        return cl.User(identifier=user, metadata={"role": user, "provider": "header"})
    else:
        with open(f"{os.getenv('JOBDIR')}/chainlit.err", "w") as f:
            f.write(f"{headers}\n")
        return None

@cl.on_message
async def on_message(message: cl.Message):

    messages = cl.user_session.get("messages")
    messages = messages + [
        {
            "content": message.content,
            "role": "user"
        }
    ]

    # Create a new message for the follow-up response
    msg = cl.Message(content="")
    await msg.send()

    # Stream the follow-up response
    stream = await client.chat.completions.create(
        messages=messages, **settings
    )

    text = ""
    async for chunk in stream:
        if token := chunk.choices[0].delta.content or "":
            text += token
            await msg.stream_token(token)

    # Add the follow-up response to message history
    messages.append(
        {"role": "assistant", "content": text}
    )
    cl.user_session.set("messages", messages)

