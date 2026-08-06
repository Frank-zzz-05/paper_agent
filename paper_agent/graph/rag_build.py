"""构建 RAG 问答图。

拓扑：
    START → retrieve → answer → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from paper_agent.graph.rag_nodes import RAGState, answer_node, retrieve_node


def build_rag_graph():
    """构建 RAG 问答图（检索 → 回答）。"""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()
