from typing import Any, Dict, List, Optional
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from langflow.custom import Component
from langflow.inputs import (
    MessageTextInput, 
    DataInput, 
    HandleInput,
    DropdownInput,
    SecretStrInput,
    FloatInput,
    IntInput,
    StrInput,
    BoolInput
)
from langflow.schema import Data
from langflow.io import Output
from langflow.field_typing import LanguageModel, Tool


# Model recommendations by provider
MODEL_RECOMMENDATIONS = {
    "openai": [
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ],
    "anthropic": [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
    ],
    "google": [
        "gemini-pro",
        "gemini-pro-vision",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    "groq": [
        "mixtral-8x7b-32768",
        "llama2-70b-4096",
        "gemma-7b-it",
    ],
    "ollama": [
        "llama2",
        "mistral",
        "codellama",
        "vicuna",
        "orca-mini",
    ],
}


class CustomToolCallingAgent(Component):
    display_name: str = "Custom Tool Calling Agent"
    description: str = "Tool calling agent with built-in LLM provider selection"
    icon = "🤖"
    name = "CustomToolCallingAgent"

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize with default provider
        self._current_provider = "openai"
        
    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None):
        """Update the build configuration dynamically based on field changes."""
        if field_name == "provider":
            # Update model_name options based on selected provider
            self._current_provider = field_value
            if field_value in MODEL_RECOMMENDATIONS:
                build_config["model_name"]["options"] = MODEL_RECOMMENDATIONS[field_value]
                # Set default to first option
                build_config["model_name"]["value"] = MODEL_RECOMMENDATIONS[field_value][0]
        return build_config

    @property
    def inputs(self):
        """Define inputs dynamically to support model recommendations."""
        # Get current provider from instance or default
        current_provider = getattr(self, '_current_provider', 'openai')
        model_options = MODEL_RECOMMENDATIONS.get(current_provider, MODEL_RECOMMENDATIONS["openai"])
        
        return [
            # LLM Configuration
            DropdownInput(
                name="provider",
                display_name="LLM Provider",
                options=["openai", "anthropic", "google", "groq", "ollama"],
                value=current_provider,
                info="Select the LLM provider to use",
                required=True,
                real_time_refresh=True,  # Enable real-time updates
            ),
            DropdownInput(
                name="model_name",
                display_name="Model Name",
                options=model_options,
                value=model_options[0] if model_options else "gpt-4-turbo",
                info="Select the model to use",
                required=True,
            ),
            SecretStrInput(
                name="api_key",
                display_name="API Key",
                info="API key for the selected provider (not needed for Ollama)",
                required=False,
            ),
            StrInput(
                name="base_url",
                display_name="Base URL",
                info="Optional: Custom base URL for the API (useful for Ollama or custom endpoints)",
                advanced=True,
                required=False,
            ),
            FloatInput(
                name="temperature",
                display_name="Temperature",
                value=0.1,
                info="Controls randomness in the model's output (0.0 = deterministic, 1.0 = creative)",
                advanced=True,
            ),
            IntInput(
                name="max_tokens",
                display_name="Max Tokens",
                value=4096,
                info="Maximum number of tokens to generate",
                advanced=True,
            ),
            
            # Agent Configuration
            MessageTextInput(
                name="system_prompt",
                display_name="System Prompt",
                info="System prompt to guide the agent's behavior",
                value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
            ),
            MessageTextInput(
                name="input_message",
                display_name="Input Message",
                info="The message to send to the agent",
                required=True,
            ),
            HandleInput(
                name="tools",
                display_name="Tools",
                input_types=["Tool"],
                is_list=True,
                info="Tools available for the agent to use",
                required=False,
            ),
            DataInput(
                name="chat_history",
                display_name="Chat History",
                is_list=True,
                advanced=True,
                info="Previous chat messages for context",
                required=False,
            ),
            IntInput(
                name="max_iterations",
                display_name="Max Iterations",
                value=10,
                info="Maximum number of iterations the agent can perform",
                advanced=True,
            ),
            FloatInput(
                name="max_execution_time",
                display_name="Max Execution Time",
                value=300.0,
                info="Maximum execution time in seconds",
                advanced=True,
            ),
            BoolInput(
                name="return_intermediate_steps",
                display_name="Return Intermediate Steps",
                value=False,
                info="Include intermediate steps in the response",
                advanced=True,
            ),
        ]

    outputs = [
        Output(display_name="Agent Response", name="response", method="run_agent"),
    ]

    def _create_llm(self) -> LanguageModel:
        """Create the appropriate LLM based on the selected provider."""
        provider = self.provider.lower()
        
        # Common parameters
        common_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            if provider == "openai":
                if not self.api_key:
                    raise ValueError("OpenAI requires an API key")
                llm = ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                    **common_params
                )
            
            elif provider == "anthropic":
                if not self.api_key:
                    raise ValueError("Anthropic requires an API key")
                llm = ChatAnthropic(
                    model=self.model_name,
                    anthropic_api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                    **common_params
                )
            
            elif provider == "google":
                if not self.api_key:
                    raise ValueError("Google requires an API key")
                llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    **common_params
                )
            
            elif provider == "groq":
                if not self.api_key:
                    raise ValueError("Groq requires an API key")
                llm = ChatGroq(
                    model=self.model_name,
                    groq_api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                    **common_params
                )
            
            elif provider == "ollama":
                # Ollama doesn't require an API key
                base_url = self.base_url if self.base_url else "http://localhost:11434"
                llm = ChatOllama(
                    model=self.model_name,
                    base_url=base_url,
                    temperature=self.temperature,
                    # Ollama doesn't use max_tokens in the same way
                )
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            return llm
            
        except Exception as e:
            raise ValueError(f"Failed to create LLM: {str(e)}")

    def _validate_tools(self):
        """Validate that tools have proper names."""
        if self.tools:
            for tool in self.tools:
                if not hasattr(tool, 'name') or not tool.name:
                    raise ValueError(f"Tool {tool} does not have a valid name")

    def _format_chat_history(self) -> List[tuple]:
        """Format chat history for the prompt template."""
        if not self.chat_history:
            return []
        
        formatted_history = []
        for msg in self.chat_history:
            if isinstance(msg, Data):
                # Extract role and content from Data object
                role = msg.data.get("role", "human")
                content = msg.data.get("content", msg.text)
                formatted_history.append((role, content))
            elif isinstance(msg, dict):
                role = msg.get("role", "human")
                content = msg.get("content", "")
                formatted_history.append((role, content))
            else:
                # Assume it's a simple string from human
                formatted_history.append(("human", str(msg)))
        
        return formatted_history

    def _escape_prompt_variables(self, prompt: str) -> str:
        """Escape template variables in the prompt to prevent parsing errors."""
        # Replace single braces with double braces, but preserve existing double braces
        # This prevents variables like {message} from being interpreted as template variables
        import re
        # First, temporarily replace existing double braces
        prompt = prompt.replace("{{", "<<DOUBLE_OPEN>>")
        prompt = prompt.replace("}}", "<<DOUBLE_CLOSE>>")
        # Then escape single braces
        prompt = prompt.replace("{", "{{")
        prompt = prompt.replace("}", "}}")
        # Finally, restore original double braces
        prompt = prompt.replace("<<DOUBLE_OPEN>>", "{{")
        prompt = prompt.replace("<<DOUBLE_CLOSE>>", "}}")
        return prompt

    def _build_agent(self) -> Any:
        """Build the tool calling agent."""
        try:
            # Create the LLM
            llm = self._create_llm()
            
            # Validate tools
            self._validate_tools()
            
            # Escape any template variables in the system prompt
            escaped_system_prompt = self._escape_prompt_variables(self.system_prompt)
            
            # Create the prompt template with explicit iteration encouragement
            iteration_guidance = (
                "\n\nIMPORTANT: If the task requires multiple steps or iterations "
                "(such as generating multiple queries, trying different approaches, or refining results), "
                "you MUST use the available tools multiple times as needed. "
                "Do not stop after a single tool use if the task requires more."
            )
            
            messages = [
                ("system", escaped_system_prompt + iteration_guidance),
            ]
            
            # Add chat history placeholder if history exists
            if self.chat_history:
                messages.append(("placeholder", "{chat_history}"))
            
            messages.extend([
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            prompt = ChatPromptTemplate.from_messages(messages)
            
            # Create the agent
            agent = create_tool_calling_agent(
                llm=llm,
                tools=self.tools or [],
                prompt=prompt
            )
            
            return agent
            
        except NotImplementedError as e:
            error_msg = (
                f"{self.provider} with model {self.model_name} does not support tool calling. "
                f"Please try using a compatible model. Common tool-calling models include:\n"
                f"- OpenAI: gpt-4-turbo, gpt-4, gpt-3.5-turbo\n"
                f"- Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku\n"
                f"- Google: gemini-pro\n"
                f"- Groq: mixtral-8x7b-32768, llama2-70b-4096"
            )
            raise NotImplementedError(error_msg) from e
        except Exception as e:
            raise ValueError(f"Failed to create agent: {str(e)}")

    def run_agent(self) -> Data:
        """Run the agent and return the response."""
        try:
            # Build the agent
            agent = self._build_agent()
            
            # Prepare the input
            agent_input = {
                "input": self.input_message,
            }
            
            # Add chat history if available
            if self.chat_history:
                agent_input["chat_history"] = self._format_chat_history()
            
            # Create an agent executor with enhanced configuration
            from langchain.agents import AgentExecutor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools or [],
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=self.max_iterations,
                max_execution_time=self.max_execution_time,
                early_stopping_method="force",  # Continue until max_iterations
                return_intermediate_steps=self.return_intermediate_steps,
            )
            
            # Run the agent
            result = agent_executor.invoke(agent_input)
            
            # Extract the output
            output = result.get("output", "No response generated")
            
            # Prepare response data
            response_data = {
                "response": output,
                "provider": self.provider,
                "model": self.model_name,
                "input": self.input_message,
            }
            
            # Add intermediate steps if requested
            if self.return_intermediate_steps and "intermediate_steps" in result:
                steps = []
                for action, observation in result["intermediate_steps"]:
                    steps.append({
                        "tool": action.tool,
                        "tool_input": action.tool_input,
                        "observation": str(observation)
                    })
                response_data["intermediate_steps"] = steps
            
            return Data(
                data=response_data,
                text=output
            )
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            return Data(
                data={"error": error_msg},
                text=error_msg
            )