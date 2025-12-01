import requests
import pandas as pd
import os
import json

API_URL = "https://api.pulsemcp.com/v0beta/servers"
DB_FILE = "db.csv"

def fetch_servers():
    """
    Fetch all servers from PulseMCP API handling pagination.
    """
    servers = []
    offset = 0
    count_per_page = 100  # Maximize page size
    
    while True:
        try:
            params = {
                "offset": offset,
                "count_per_page": count_per_page
            }
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            batch = data.get("servers", [])
            if not batch:
                break
                
            servers.extend(batch)
            
            # Check if we need to fetch more
            total_count = data.get("total_count", 0)
            if len(servers) >= total_count:
                break
                
            offset += len(batch)
            print(f"Fetched {len(servers)} / {total_count} servers...")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    return servers

def is_remote(server):
    """
    Check if the server has a remote connection (HTTP/SSE).
    """
    remotes = server.get("remotes", [])
    for remote in remotes:
        if remote.get("url_direct") or remote.get("url_setup"):
            return True
    return False

def get_auth_type(server):
    """
    Extract auth type from the first remote.
    """
    remotes = server.get("remotes", [])
    if not remotes:
        return "none"
    
    # Prioritize the first remote's auth method
    auth = remotes[0].get("authentication_method", "none")
    
    # Map to CSV conventions
    if auth == "api_key":
        return "api-key"
    elif auth == "oauth":
        return "oauth"
    elif not auth or auth == "null":
        return "none"
    return auth

def get_maturity(server):
    """
    Infer maturity from GitHub stars.
    """
    stars = server.get("github_stars", 0)
    if stars is None:
        stars = 0
    return "stable" if stars > 1000 else "beta"

def extract_capabilities(description):
    """
    Extract simple keywords as capability tags.
    """
    if not description:
        return ""
    
    keywords = ["database", "sql", "image", "vision", "browser", "search", "pdf", "docker", "cloud", "aws", "gcp", "azure"]
    found = [k for k in keywords if k in description.lower()]
    return "\n".join(found)

def main():
    print("Starting PulseMCP crawler...")
    
    # 1. Fetch Data
    all_servers = fetch_servers()
    print(f"Total servers fetched: {len(all_servers)}")
    
    # 2. Filter for Remote
    remote_servers = [s for s in all_servers if is_remote(s)]
    print(f"Remote servers found: {len(remote_servers)}")
    
    if not remote_servers:
        print("No remote servers found. Exiting.")
        return

    # 3. Prepare New Data
    new_rows = []
    for s in remote_servers:
        server_name = s.get("name")
        if not server_name:
            continue
            
        row = {
            "server_name": server_name,
            "server_url": s.get("url") or s.get("external_url") or "",
            "server_description": s.get("short_description") or "",
            "auth_type": get_auth_type(s),
            "maturity": get_maturity(s),
            "compatability": "cursor\nclaude_desktop", # Default
            "tool_name": server_name, # Placeholder as per plan
            "tool_description": s.get("short_description") or "",
            "example_queries": "", # Empty as per user request
            "capability_tags": extract_capabilities(s.get("short_description")),
            "embedded_text": "" # Empty as per user request
        }
        new_rows.append(row)
        
    new_df = pd.DataFrame(new_rows)
    
    # 4. Update CSV
    if os.path.exists(DB_FILE):
        print(f"Reading existing {DB_FILE}...")
        try:
            existing_df = pd.read_csv(DB_FILE)
            
            # Deduplicate: Remove rows from new_df that already exist in existing_df (by server_name)
            # We want to preserve existing manual edits, so we prioritize existing_df
            existing_servers = set(existing_df["server_name"].unique())
            
            # Filter out servers that are already in the DB
            # Note: This checks server_name. If a server has multiple tools in DB, it's still "in DB".
            # Since we only add 1 row per server (generic tool), this is safe.
            filtered_new_df = new_df[~new_df["server_name"].isin(existing_servers)]
            
            print(f"New servers to add: {len(filtered_new_df)}")
            
            if not filtered_new_df.empty:
                final_df = pd.concat([existing_df, filtered_new_df], ignore_index=True)
                final_df.to_csv(DB_FILE, index=False)
                print(f"Successfully updated {DB_FILE}. Total rows: {len(final_df)}")
            else:
                print("No new servers to add.")
                
        except Exception as e:
            print(f"Error reading/writing CSV: {e}")
    else:
        print(f"Creating new {DB_FILE}...")
        new_df.to_csv(DB_FILE, index=False)
        print(f"Created {DB_FILE} with {len(new_df)} rows.")

if __name__ == "__main__":
    main()
