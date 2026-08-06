"""图构建：编译并返回可执行的 LangGraph。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from paper_agent.graph.nodes import extract, finalize, load_documents, summarize
from paper_agent.graph.state import AgentState


def build_paper_graph():
    """构建论文读取图。

    拓扑（并行分支，节点自守卫）：
        START → load_documents →┬→ summarize ─┐
                                └→ extract   ─┴→ finalize → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_documents", load_documents)
    graph.add_node("summarize", summarize)
    graph.add_node("extract", extract)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "load_documents")
    graph.add_edge("load_documents", "summarize")
    graph.add_edge("load_documents", "extract")
    graph.add_edge("summarize", "finalize")
    graph.add_edge("extract", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
