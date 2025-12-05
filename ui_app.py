from __future__ import annotations

from typing import List

import streamlit as st

from mcp_suggester.server import suggest_mcp_servers_impl

st.set_page_config(page_title="MCP Finder", page_icon=None)
st.title("MCP Finder")
st.write(
    "Enter a task in natural language and this app will recommend the most relevant "
    "Model Context Protocol servers and tools."
)


def parse_tags(tag_input: str) -> List[str]:
    return [tag.strip() for tag in tag_input.split(",") if tag.strip()]


with st.form("recommendation_form"):
    user_query = st.text_area(
        "User query",
        placeholder="Summarize the latest EU AI Act updates from trusted sources.",
    )
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Servers to return", min_value=1, max_value=5, value=3)
    with col2:
        max_candidates = st.slider("Candidate tools to examine", min_value=5, max_value=40, value=20, step=5)
    filter_tags_input = st.text_input(
        "Optional filter tags (comma separated)",
        placeholder="security,docker",
    )
    submitted = st.form_submit_button("Suggest MCP servers")

if submitted:
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        tags = parse_tags(filter_tags_input)
        with st.spinner("Finding relevant MCP servers..."):
            suggestions = suggest_mcp_servers_impl(
                user_query=user_query,
                top_n=top_n,
                max_candidates=max_candidates,
                filter_tags=tags or None,
            )
        if not suggestions or suggestions[0]["server_name"] == "none":
            st.error("No matching servers found.")
        else:
            for suggestion in suggestions:
                with st.expander(
                    f"{suggestion['server_name']} ({suggestion.get('score', 0):.2f})",
                    expanded=True,
                ):
                    st.write(f"**Server URL:** {suggestion.get('server_url', '')}")
                    st.write(f"**Auth type:** {suggestion.get('auth_type', '')}")
                    st.write(f"**Reason:** {suggestion.get('reason', '')}")
                    for tool in suggestion.get("tools", []):
                        st.markdown(
                            f"- **{tool['tool_name']}** (score {tool.get('score', 0):.2f})  \n"
                            f"  {tool.get('reason', '')}"
                        )
                        example = tool.get("example_query_to_run")
                        if example:
                            st.code(example, language="text")
