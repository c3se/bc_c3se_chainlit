import os
import pwd
import traceback
from typing import Dict, List, Any, Optional

import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent

client = AsyncOpenAI(base_url=os.getenv("VLLM_BASE_URL"), api_key=os.getenv("VLLM_API_KEY"))

cl.instrument_openai()

mcp_tools_cache = {}

@cl.on_chat_start
async def start():
    username = os.getenv("USER")
    fullname = pwd.getpwnam(username).pw_gecos.split(',')[0]

    model = os.getenv("HF_MODEL")
    # find the exact model name for local model snapshot
    if model.startswith("/mimer"):
        for dirname in model.split("/"):
            # trim "models--" prefix for models from HF
            modelname = ""
            if dirname.startswith("models"):
                modelname = dirname[8:]
                break
    modelname = modelname or model

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                items={modelname: model},
                initial_value=model,
            ),
            Switch(id="stream", label="Stream Tokens", initial=True),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.8,
                min=0,
                max=1,
                step=0.1,
            ),
        ]
    ).send()

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

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    cl.Message(f"Connected to MCP server: {connection.name}").send()

    try:
        result = await session.list_tools()

        tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

        mcp_tools_cache[connection.name] = tools

        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)

        await cl.Message(
            f"Found {len(tools)} tools from {connection.name} MCP server."
        ).send()
    except Exception as e:
        await cl.Message(f"Error listing tools from MCP server: {str(e)}").send()


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    if name in mcp_tools_cache:
        del mcp_tools_cache[name]

    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)

    await cl.Message(f"Disconnected from MCP server: {name}").send()

@cl.step(type="tool")
async def execute_tool(tool_name: str, tool_input: Dict[str, Any]):
    print("Executing tool:", tool_name)
    print("Tool input:", tool_input)
    mcp_name = None
    mcp_tools = cl.user_session.get("mcp_tools", {})

    for conn_name, tools in mcp_tools.items():
        if any(tool["name"] == tool_name for tool in tools):
            mcp_name = conn_name
            break

    if not mcp_name:
        return {"error": f"Tool '{tool_name}' not found in any connected MCP server"}

    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)

    try:
        result = await mcp_session.call_tool(tool_name, tool_input)
        return result
    except Exception as e:
        return {"error": f"Error calling tool '{tool_name}': {str(e)}"}


async def format_tools_for_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    openai_tools = []

    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools

def format_calltoolresult_content(result):
    """Extract text content from a CallToolResult object.

    The MCP CallToolResult contains a list of content items,
    where we want to extract text from TextContent type items.
    """
    text_contents = []

    if isinstance(result, CallToolResult):
        for content_item in result.content:
            # This script only supports TextContent but you can implement other CallToolResult types
            if isinstance(content_item, TextContent):
                text_contents.append(content_item.text)

    if text_contents:
        return "\n".join(text_contents)
    return str(result)

@cl.on_message
async def on_message(message: cl.Message):
    try:
        messages = cl.user_session.get("messages")
        settings = cl.user_session.get("chat_settings")

        messages.append({"role": "user", "content": message.content})

        # Initial message for the first assistant response
        initial_msg = cl.Message(content="")
        await initial_msg.send()

        mcp_tools = cl.user_session.get("mcp_tools", {})
        all_tools = []
        for connection_tools in mcp_tools.values():
            all_tools.extend(connection_tools)

        chat_params = {**settings}
        if all_tools:
            openai_tools = await format_tools_for_openai(all_tools)
            chat_params["tools"] = openai_tools
            chat_params["tool_choice"] = "auto"

        stream = await client.chat.completions.create(
            messages=messages, **chat_params
        )

        initial_response = ""
        tool_calls = []

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if token := delta.content or "":
                initial_response += token
                await initial_msg.stream_token(token)

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tc_id = tool_call.index
                    if tc_id >= len(tool_calls):
                        tool_calls.append({"name": "", "arguments": ""})

                    if tool_call.function.name:
                        tool_calls[tc_id]["name"] = tool_call.function.name

                    if tool_call.function.arguments:
                        tool_calls[tc_id]["arguments"] += tool_call.function.arguments

        # First, update message history with the initial response
        if initial_response.strip():
            messages.append({"role": "assistant", "content": initial_response})

        # Process tool calls if any
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                try:
                    import json

                    tool_args = json.loads(tool_call["arguments"])

                    # Add the tool call to message history
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{len(messages)}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call["arguments"],
                                    },
                                }
                            ],
                        }
                    )

                    # Execute the tool in a step
                    with cl.Step(name=f"Executing tool: {tool_name}", type="tool"):
                        tool_result = await execute_tool(tool_name, tool_args)

                    # Format the tool result content
                    tool_result_content = format_calltoolresult_content(tool_result)

                    # Display the tool result to the user
                    tool_result_msg = cl.Message(
                        content=f"Tool Result from {tool_name}:\n{tool_result_content}",
                        author="Tool",
                    )
                    await tool_result_msg.send()

                    # Add the tool result to message history
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{len(messages)-1}",
                            "content": tool_result_content,
                        }
                    )

                    # Create a new message for the follow-up response
                    follow_up_msg = cl.Message(content="")
                    await follow_up_msg.send()

                    # Stream the follow-up response
                    follow_up_stream = await client.chat.completions.create(
                        messages=messages, **settings
                    )

                    follow_up_text = ""
                    async for chunk in follow_up_stream:
                        if token := chunk.choices[0].delta.content or "":
                            follow_up_text += token
                            await follow_up_msg.stream_token(token)

                    # Add the follow-up response to message history
                    messages.append(
                        {"role": "assistant", "content": follow_up_text}
                    )

                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    error_message = cl.Message(content=error_msg)
                    await error_message.send()

        # Update the session message history
        cl.user_session.set("messages", messages)

    except:
        error_message = traceback.format_exc()
        await cl.Message(content=error_message).send()

