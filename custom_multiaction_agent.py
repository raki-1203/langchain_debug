import os

from typing import List, Tuple, Any, Union

from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.schema import AgentAction, AgentFinish

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY


def random_word(query: str) -> str:
    print("\nNow I'm doing this!")
    return 'foo'


class FakeAgent(BaseMultiActionAgent):
    """Fake Custom Agent."""

    @property
    def input_keys(self):
        return ['input']

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if len(intermediate_steps) == 0:
            return [
                AgentAction(tool='Search', tool_input=kwargs['input'], log=''),
                AgentAction(tool='RandomWord', tool_input=kwargs['input'], log=''),
            ]
        else:
            return AgentFinish(return_values={'output': 'bar'}, log='')

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Actoin specifying what tool to use.
        """
        if len(intermediate_steps) == 0:
            return [
                AgentAction(tool='Search', tool_input=kwargs['input'], log=''),
                AgentAction(tool='RandomWord', tool_input=kwargs['input'], log=''),
            ]
        else:
            return AgentFinish(return_values={'output': 'bar'}, log='')


if __name__ == '__main__':
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name='Search',
            func=search.run,
            description='useful for when you need to answer questions about current events',
        ),
        Tool(
            name='RandomWord',
            func=random_word,
            description='call this to get a random word.',
        ),
    ]

    agent = FakeAgent()
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    agent_executor.run('How many people live in canada as of 2023?')


