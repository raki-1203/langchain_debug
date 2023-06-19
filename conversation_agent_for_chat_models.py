import os

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY


if __name__ == '__main__':
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name='Search',
            func=search.run,
            description='useful for when you need to answer questions about current events or the current state of '
                        'the world. the input to this should be a single search term.',
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatOpenAI(temperature=0)

    agent_chain = initialize_agent(tools,
                                   llm,
                                   agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                   verbose=True,
                                   memory=memory)

    agent_chain.run(input='hi, i am raki')
    agent_chain.run(input="what's my name?")
    agent_chain.run("what are some good dinners to make this week, if i like thai food?")
    agent_chain.run(input="tell me the last letter in my name, and also tell me who won the world cup in 1978?")
    agent_chain.run(input="whats the weather like in pomfret?")


