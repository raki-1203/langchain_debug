import os
import faiss

from collections import deque
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from langchain import LLMChain, OpenAI, PromptTemplate, SerpAPIWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['SERPER_API_KEY'] = c.SERPER_API_KEY
os.environ['GOOGLE_API_KEY'] = c.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = c.GOOGLE_CSE_ID


if __name__ == '__main__':
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()

    # initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. "
        "Come up with a todo list for this objective: {objective}"
    )
    todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name='Search',
            func=search.run,
            description='useful for when you need to answer questions about current events',
        ),
        Tool(
            name='TODO',
            func=todo_chain.run,
            description="useful for when you need to come up with todo lists. "
                        "Input: an objective to create a todo list for. "
                        "Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
    ]

    prefix = """You are an AI who performs one task based on the following objetive:
{objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
{agent_scratchpad}"""
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=['objective', 'context', 'task', 'agent_scratchpad'],
    )

    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    # Run the BabyAGI
    OBJECTIVE = "Organize Tesla's car lineup in an easy-to-understand way even for middle school students"

    # Logging of LLMChains
    verbose = False
    # If None, will keep on going forever
    max_iterations: Optional[int] = 5
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose,
        max_iterations=max_iterations,
    )

    print(baby_agi({"objective": OBJECTIVE}))



