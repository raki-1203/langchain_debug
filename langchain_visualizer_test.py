import os
import re
import langchain_visualizer
import asyncio

from typing import List, Union

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain, LLMMathChain
from langchain.schema import AgentAction, AgentFinish

import config as c

os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY


# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=OpenAI(temperature=0), verbose=True)

tools = [
    Tool(
        name='Search',
        func=search.run,
        description='useful for when you need to answer questions about current events',
    ),
    Tool(
        name='Calculator',
        func=llm_math_chain.run,
        description='useful for when you need to answer questions about math',
    ),
]


# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. 
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop('intermediate_steps')
        thoughts = ''
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f'\nObservation: {observation}\nThought: '
        # Set the agent_scratchpad variable to that value
        kwargs['agent_scratchpad'] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs['tools'] = '\n'.join([f'{tool.name}: {tool.description}' for tool in self.tools])
        # Create a list of tool names for the tools proviced
        kwargs['tool_names'] = ', '.join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={'output': llm_output.split('Final Answer:')[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(' ').strip('"'), log=llm_output)


if __name__ == '__main__':
    # Prompt Template
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=['input', 'intermediate_steps'],
    )
    # print(prompt)

    # Output Parser
    output_parser = CustomOutputParser()
    # print(output_parser)

    # Set up LLM
    llm = OpenAI(temperature=0)

    # Set up the Agent
    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=['\nObservation:'],  # Define the stop sequence
        allowed_tools=tool_names,
    )

    # Use the Agent
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    async def run_agent_demo():
        return agent_executor.run("What if you multiply the age of the person Lee Seung-gi married by 3 and add 2023?")

    langchain_visualizer.visualize(run_agent_demo)





