# AgenticGrader Cookbook

This directory contains usage examples for AgenticGrader, demonstrating how to use tool-augmented LLMs for fact-checking and evaluation.

## Overview

AgenticGrader is a core component in OpenJudge for tool-augmented evaluation. It allows LLMs to autonomously call tools (such as web search, code execution, etc.) during the evaluation process to gather information and make more accurate judgments.

## Examples

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [01_native_react_native_tool.py](./01_native_react_native_tool.py) | Built-in ReActAgent + Native Tool | `tavily-python` |
| [02_native_react_langchain_tool.py](./02_native_react_langchain_tool.py) | Built-in ReActAgent + LangChain Tool | `langchain-tavily` |
| [03_langchain_agent.py](./03_langchain_agent.py) | LangChain Agent (Full Delegation) | `langchain`, `langchain-openai`, `langchain-tavily` |
| [04_agentscope_agent.py](./04_agentscope_agent.py) | AgentScope Agent (Full Delegation) | `agentscope`, `tavily-python` |

## Architecture

AgenticGrader supports four different combination approaches:

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgenticGrader                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Approach 1: Built-in ReActAgent + Native Tool (Zero Deps)      │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │ ReActAgent  │───▶│ BaseTool    │                             │
│  │  (built-in) │    │  (custom)   │                             │
│  └─────────────┘    └─────────────┘                             │
│                                                                 │
│  Approach 2: Built-in ReActAgent + LangChain Tool               │
│  ┌─────────────┐    ┌─────────────────────┐                     │
│  │ ReActAgent  │───▶│ LangChainToolAdapter│                     │
│  │  (built-in) │    │  (wraps LC tools)   │                     │
│  └─────────────┘    └─────────────────────┘                     │
│                                                                 │
│  Approach 3: LangChain Agent (Full Delegation)                  │
│  ┌─────────────────────┐                                        │
│  │ LangChainAgentAdapter│                                       │
│  │  (wraps LC Agent)   │                                        │
│  └─────────────────────┘                                        │
│                                                                 │
│  Approach 4: AgentScope Agent (Full Delegation)                 │
│  ┌───────────────────────┐                                      │
│  │ AgentScopeAgentAdapter│                                      │
│  │  (wraps AS Agent)     │                                      │
│  └───────────────────────┘                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Environment Variables

Before running the examples, make sure to set the following environment variables:

```bash
# OpenAI-compatible API Key (required)
export OPENAI_API_KEY="your-api-key"

# Tavily Search API Key (for web search)
export TAVILY_API_KEY="your-tavily-api-key"
```

## Quick Start

```bash
# Install basic dependencies
pip install openjudge tavily-python

# Run the simplest example
python 01_native_react_native_tool.py
```

## Selection Guide

| Scenario | Recommended Approach |
|----------|---------------------|
| No third-party framework dependencies | Approach 1: Built-in ReActAgent + Native Tool |
| Want to use LangChain's rich tool ecosystem | Approach 2: Built-in ReActAgent + LangChain Tool |
| Have existing LangChain Agent to reuse | Approach 3: LangChain Agent |
| Have existing AgentScope Agent to reuse | Approach 4: AgentScope Agent |
