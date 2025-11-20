import requests
import time
import json
from collections import defaultdict, deque
from typing import Dict, List
import numpy as np
from datetime import datetime
import re

# Check for BeautifulSoup
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: BeautifulSoup not found. Please run: pip install beautifulsoup4")
    exit(1)

class WebCrawler:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.graph = defaultdict(list)        # Outgoing links: page -> [link1, link2]
        self.reverse_graph = defaultdict(list) # Incoming links: link -> [page1, page2]
        self.page_data = {}                   # Stores node_id, last_updated, etc.
        self.visited = set()
        self.first_visit_time = None
        self.last_eval_time = None
        self.visit_count = 0
        self.all_pages = set()

    def fetch_page(self, page_id: str) -> Dict:
        """Fetches HTML, scrapes Node ID, Links, and History."""
        url = f"{self.base_url}/{page_id}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            self.visit_count += 1
            
            if self.first_visit_time is None:
                self.first_visit_time = time.time()

            soup = BeautifulSoup(response.text, 'html.parser')
            data = {}
            
            # Scrape Node ID
            node_id_tag = soup.find('span', class_='node-id')
            if node_id_tag:
                full_text = node_id_tag.get_text(strip=True)
                if "Node ID:" in full_text:
                    data['node_id'] = full_text.replace("Node ID:", "").strip()

            # Scrape Last Updated
            last_updated_tag = soup.find('span', class_='last-updated')
            if last_updated_tag:
                txt = last_updated_tag.string or ""
                data['last_updated'] = txt.replace('Last Updated: ', '').strip()

            # Scrape Links
            links = []
            link_tags = soup.find_all('a', class_='file-link')
            for link in link_tags:
                href = link.get('href')
                if href:
                    link_id = href.strip('/')
                    if link_id:
                        links.append(link_id)
            data['links'] = links
            
            # Scrape History (optional but good for parsing context)
            history = []
            history_summary_tag = soup.find('summary')
            if history_summary_tag:
                history_container = history_summary_tag.find_next_sibling('div')
                if history_container:
                    history_items = history_container.find_all('div')
                    for item in history_items:
                        text = item.get_text(strip=True)
                        match = re.search(r'•\s*([\w\d]+)\s*\((.*?)\s*UTC\)', text)
                        if match:
                            history.append({
                                'node_id': match.group(1),
                                'timestamp': match.group(2)
                            })
            data['history'] = history

            if 'node_id' not in data:
                 return None

            return data
            
        except Exception:
            return None
    
    def process_page(self, page_id: str, data: Dict):
        """Updates the graph structure and page data cache."""
        if data is None:
            return
        
        self.page_data[page_id] = {
            'node_id': data.get('node_id'),
            'last_updated': data.get('last_updated'),
            'history': data.get('history', []),
            'fetch_time': time.time()
        }
        
        outgoing = data.get('links', [])
        self.graph[page_id] = outgoing
        self.all_pages.add(page_id)
        self.all_pages.update(outgoing)
        
        for link in outgoing:
            self.reverse_graph[link].append(page_id)
        
        self.visited.add(page_id)
    
    def calculate_pagerank(self, damping=0.85, iterations=100, tolerance=1e-6) -> Dict[str, float]:
        """Computes PageRank using the power iteration method."""
        if not self.all_pages:
            return {}
        
        pages = list(self.all_pages)
        n = len(pages)
        page_to_idx = {page: idx for idx, page in enumerate(pages)}
        
        pr = np.ones(n) / n
        
        for _ in range(iterations):
            new_pr = np.ones(n) * (1 - damping) / n
            
            for page in pages:
                idx = page_to_idx[page]
                incoming = self.reverse_graph.get(page, [])
                
                for incoming_page in incoming:
                    if incoming_page in page_to_idx:
                        incoming_idx = page_to_idx[incoming_page]
                        outgoing_count = len(self.graph.get(incoming_page, []))
                        
                        if outgoing_count > 0:
                            new_pr[idx] += damping * pr[incoming_idx] / outgoing_count
            
            dangling_sum = 0
            for page in pages:
                idx = page_to_idx[page]
                if not self.graph.get(page, []):
                    dangling_sum += pr[idx]
            
            new_pr += damping * dangling_sum / n
            
            if np.sum(np.abs(new_pr - pr)) < tolerance:
                break
            pr = new_pr
        
        pr = pr / np.sum(pr)
        return {page: pr[page_to_idx[page]] for page in pages}
    
    def prioritized_crawl(self, start_page="page_0", time_budget=55, initial_queue: List[str] = None):
        """
        Executes a 3-phase crawl strategy:
        1. Discovery (BFS)
        2. Expansion (Highest In-Degree)
        3. Maintenance (Re-visiting stale pages with high PageRank)
        """
        if initial_queue:
            queue = deque(initial_queue)
        else:
            queue = deque([start_page])
        
        start_time = time.time()
        
        # Phase 1: Initial BFS (Fast Discovery) - 8 seconds
        phase1_limit = 8
        while queue and (time.time() - start_time) < phase1_limit:
            page_id = queue.popleft()
            if page_id in self.visited:
                continue
            
            data = self.fetch_page(page_id)
            if data:
                self.process_page(page_id, data)
                for link in data.get('links', []):
                    if link not in self.visited:
                        queue.append(link)
        
        # Phase 2: Targeted High-In-Degree Crawl - 18 seconds total
        phase2_limit = 18
        while (time.time() - start_time) < phase2_limit:
            unvisited_priority = []
            for page in self.all_pages:
                if page not in self.visited:
                    incoming_count = len(self.reverse_graph.get(page, []))
                    unvisited_priority.append((incoming_count, page))
            
            if not unvisited_priority:
                break
            
            unvisited_priority.sort(reverse=True)
            
            # Fetch top 5 candidates
            for _, page_id in unvisited_priority[:5]:
                if (time.time() - start_time) >= phase2_limit:
                    break
                if page_id in self.visited:
                    continue
                
                data = self.fetch_page(page_id)
                if data:
                    self.process_page(page_id, data)
        
        # Phase 3: Refresh Loop (Handling Staleness)
        refresh_interval = 3
        last_refresh = time.time()
        
        while (time.time() - start_time) < time_budget:
            current_time = time.time()
            
            if current_time - last_refresh >= refresh_interval:
                pagerank = self.calculate_pagerank()
                
                refresh_candidates = []
                for page_id in self.visited:
                    if page_id in self.page_data:
                        staleness = current_time - self.page_data[page_id]['fetch_time']
                        pr_score = pagerank.get(page_id, 0)
                        # Score = Importance * Staleness
                        priority = pr_score * staleness
                        refresh_candidates.append((priority, page_id))
                
                refresh_candidates.sort(reverse=True)
                
                # Refresh top 3 candidates
                refresh_count = min(3, len(refresh_candidates))
                for _, page_id in refresh_candidates[:refresh_count]:
                    if (time.time() - start_time) >= time_budget:
                        break
                    
                    data = self.fetch_page(page_id)
                    if data:
                        self.process_page(page_id, data)
                
                last_refresh = current_time
            
            time.sleep(0.5)
    
    def submit_evaluation(self) -> bool:
        """Submits current estimates to the evaluation server."""
        pagerank = self.calculate_pagerank()
        
        entries = []
        for page_id in self.visited:
            if page_id in self.page_data and page_id in pagerank:
                entries.append({
                    "page_id": page_id,
                    "latest_node_id": self.page_data[page_id]['node_id'],
                    "score": float(pagerank[page_id])
                })
        
        if not entries:
            return True
        
        url = f"{self.base_url}/evaluate"
        try:
            response = requests.post(url, json={"entries": entries}, timeout=5)
            response.raise_for_status()
            result = response.json()
            self.last_eval_time = time.time()
            
            # We keep these prints because they are the actual result of the assignment
            print(f"Eval: MSE={result.get('mse', 0):.5f}, Cov={result.get('coverage', 0):.1%}, "
                  f"Stale={result.get('avg_staleness', 0):.1f}ms, Matches={result.get('matched_entries')}")
            
            return True 
            
        except requests.exceptions.HTTPError as e:
            try:
                error_msg = e.response.json().get("error", "").lower()
                if "evaluation window" in error_msg or "evaluation has ended" in error_msg:
                    print("Evaluation window ended.")
                    return False
            except:
                pass
            return True
        except Exception as e:
            print(f"Submission Error: {e}")
            return True 
    
    def run(self, start_page="page_0"):
        """Main execution loop."""
        print(f"Crawler started at {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        
        # 1. Bootstrap from Portal
        portal_data = self.fetch_page(start_page)
        if not portal_data or not portal_data['links']:
            print("Error: Could not bootstrap from portal.")
            self.submit_evaluation()
            return
            
        initial_pages = portal_data['links']
        actual_start_page = initial_pages[0]

        # 2. Initial Crawl (7 seconds)
        self.prioritized_crawl(actual_start_page, time_budget=7, initial_queue=initial_pages)

        if self.first_visit_time is None:
            print("Error: Failed to visit any pages.")
            return

        if not self.submit_evaluation():
             return 

        # 3. Main Loop (Crawl -> Evaluate -> Repeat)
        eval_interval = 12 
        crawl_duration = eval_interval - 2 
        
        while True:
            self.prioritized_crawl(actual_start_page, time_budget=crawl_duration)
            
            if not self.submit_evaluation():
                break
                
            if (self.first_visit_time is not None) and (time.time() - self.first_visit_time) > 70:
                 break
        
        # Final submission
        self.submit_evaluation()
        print(f"Finished. Visits: {self.visit_count}, Discovered: {len(self.all_pages)}")

if __name__ == "__main__":
    BASE_URL = "http://localhost:3000"
    PORTAL_ID = None

    # Auto-discover Portal ID
    try:
        response = requests.get(BASE_URL + "/", timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        portal_id_tag = soup.find('div', class_='page-id')
        if portal_id_tag:
            full_text = portal_id_tag.get_text(strip=True)
            if "Page ID:" in full_text:
                PORTAL_ID = full_text.replace("Page ID:", "").strip()
                print(f"Discovered Portal ID: {PORTAL_ID}")
    except Exception:
        print("Error: Could not connect to server.")
        exit(1)
    
    if PORTAL_ID:
        crawler = WebCrawler(base_url=BASE_URL)
        crawler.run(start_page=PORTAL_ID)
    else:
        print("Error: Could not find Portal ID.")