"""`python -m arxiv_mcp` 启动 MCP server。"""

from arxiv_mcp.server import mcp

if __name__ == "__main__":
    mcp.run()
