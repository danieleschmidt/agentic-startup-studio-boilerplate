#!/usr/bin/env python3
"""
Repository Hygiene Bot
Automates repository maintenance tasks according to the hygiene checklist.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
from dataclasses import dataclass


@dataclass
class RepoInfo:
    """Repository information."""
    name: str
    full_name: str
    description: Optional[str]
    homepage: Optional[str]
    topics: List[str]
    archived: bool
    fork: bool
    template: bool
    last_pushed: datetime
    stargazers_count: int
    owner: str


class GitHubClient:
    """GitHub API client for repository operations."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN environment variable.")
        
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'repo-hygiene-bot/1.0'
        }
        self.base_url = 'https://api.github.com'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to GitHub API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, json=data)
            elif method.upper() == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json() if response.content else {}
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def list_user_repos(self, per_page: int = 100) -> List[RepoInfo]:
        """Step 0: List user repositories with filtering."""
        self.logger.info("Fetching user repositories...")
        
        repos = []
        page = 1
        
        while True:
            endpoint = f"user/repos?per_page={per_page}&page={page}&affiliation=owner"
            response = self._make_request('GET', endpoint)
            
            if not response:
                break
            
            for repo_data in response:
                # Skip forks, templates, and archived repos
                if repo_data['fork'] or repo_data.get('is_template', False) or repo_data['archived']:
                    continue
                
                repo = RepoInfo(
                    name=repo_data['name'],
                    full_name=repo_data['full_name'],
                    description=repo_data.get('description'),
                    homepage=repo_data.get('homepage'),
                    topics=repo_data.get('topics', []),
                    archived=repo_data['archived'],
                    fork=repo_data['fork'],
                    template=repo_data.get('is_template', False),
                    last_pushed=datetime.fromisoformat(repo_data['pushed_at'].replace('Z', '+00:00')),
                    stargazers_count=repo_data['stargazers_count'],
                    owner=repo_data['owner']['login']
                )
                repos.append(repo)
            
            if len(response) < per_page:
                break
            page += 1
        
        self.logger.info(f"Found {len(repos)} eligible repositories")
        return repos
    
    def update_repo_metadata(self, repo: RepoInfo, description: str = None, 
                           homepage: str = None, topics: List[str] = None) -> bool:
        """Step 1: Update repository description, website & topics."""
        changes_made = False
        
        # Update description and homepage
        if description or homepage:
            update_data = {}
            if description and description != repo.description:
                update_data['description'] = description
                changes_made = True
            
            if homepage and homepage != repo.homepage:
                update_data['homepage'] = homepage
                changes_made = True
            
            if update_data:
                self._make_request('PATCH', f"repos/{repo.full_name}", update_data)
                self.logger.info(f"Updated metadata for {repo.name}")
        
        # Update topics
        if topics and set(topics) != set(repo.topics):
            topic_data = {'names': topics}
            self._make_request('PUT', f"repos/{repo.full_name}/topics", topic_data)
            self.logger.info(f"Updated topics for {repo.name}")
            changes_made = True
        
        return changes_made
    
    def create_pull_request(self, repo_name: str, title: str, body: str, 
                          head_branch: str, base_branch: str = 'main') -> str:
        """Step 10: Create pull request."""
        pr_data = {
            'title': title,
            'body': body,
            'head': head_branch,
            'base': base_branch
        }
        
        response = self._make_request('POST', f"repos/{repo_name}/pulls", pr_data)
        
        # Add label and assignee
        pr_number = response['number']
        
        # Add label
        label_data = {'labels': ['automated-maintenance']}
        self._make_request('POST', f"repos/{repo_name}/issues/{pr_number}/labels", label_data)
        
        # Add assignee
        assignee_data = {'assignees': ['danieleschmidt']}
        self._make_request('POST', f"repos/{repo_name}/issues/{pr_number}/assignees", assignee_data)
        
        return response['html_url']
    
    def pin_repositories(self, repo_names: List[str]) -> bool:
        """Step 8: Pin top repositories by star count."""
        pin_data = {'repository_ids': repo_names[:6]}  # GitHub allows max 6 pinned repos
        
        try:
            self._make_request('PUT', 'user/pinned_repositories', pin_data)
            self.logger.info(f"Pinned {len(pin_data['repository_ids'])} repositories")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pin repositories: {e}")
            return False
    
    def archive_repository(self, repo_name: str) -> bool:
        """Step 6: Archive stale repository."""
        try:
            self._make_request('PATCH', f"repos/{repo_name}", {'archived': True})
            self.logger.info(f"Archived repository: {repo_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to archive {repo_name}: {e}")
            return False


class RepoHygieneBot:
    """Main repository hygiene automation bot."""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github = GitHubClient(github_token)
        self.logger = logging.getLogger(__name__)
        self.changes_made = []
    
    def generate_description(self, repo_name: str) -> str:
        """Generate a concise description for repositories missing one."""
        # Basic heuristics based on repository name
        name_lower = repo_name.lower()
        
        if 'bot' in name_lower:
            return f"Automated bot for {name_lower.replace('bot', '').replace('-', ' ').strip()}"
        elif 'api' in name_lower:
            return f"API service for {name_lower.replace('api', '').replace('-', ' ').strip()}"
        elif 'frontend' in name_lower or 'ui' in name_lower:
            return f"Frontend application for {name_lower.replace('frontend', '').replace('ui', '').replace('-', ' ').strip()}"
        elif 'template' in name_lower:
            return f"Template for {name_lower.replace('template', '').replace('-', ' ').strip()}"
        else:
            return f"Repository for {name_lower.replace('-', ' ')}"
    
    def suggest_topics(self, repo: RepoInfo) -> List[str]:
        """Suggest relevant topics based on repository characteristics."""
        suggested = set(repo.topics)  # Keep existing topics
        name_lower = repo.name.lower()
        desc_lower = (repo.description or '').lower()
        
        # Technology stack detection
        if any(term in name_lower + desc_lower for term in ['python', 'py']):
            suggested.add('python')
        if any(term in name_lower + desc_lower for term in ['javascript', 'js', 'node']):
            suggested.add('javascript')
        if any(term in name_lower + desc_lower for term in ['typescript', 'ts']):
            suggested.add('typescript')
        if any(term in name_lower + desc_lower for term in ['react', 'nextjs']):
            suggested.add('react')
        if any(term in name_lower + desc_lower for term in ['api', 'fastapi', 'flask']):
            suggested.add('api')
        
        # Purpose-based topics
        if any(term in name_lower + desc_lower for term in ['automation', 'bot']):
            suggested.add('automation')
        if any(term in name_lower + desc_lower for term in ['template', 'boilerplate']):
            suggested.add('template')
        if any(term in name_lower + desc_lower for term in ['security', 'scanner']):
            suggested.add('security')
        
        # Ensure at least 5 topics
        while len(suggested) < 5:
            default_topics = ['opensource', 'github-actions', 'devops', 'productivity', 'tool']
            for topic in default_topics:
                if topic not in suggested:
                    suggested.add(topic)
                    break
            else:
                break
        
        return list(suggested)
    
    def run_hygiene_check(self, repo_name: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete hygiene checklist."""
        self.logger.info("Starting repository hygiene check...")
        
        # Step 0: List repositories
        repos = self.github.list_user_repos()
        
        if repo_name:
            repos = [r for r in repos if r.name == repo_name]
            if not repos:
                raise ValueError(f"Repository {repo_name} not found or not eligible")
        
        results = {
            'processed_repos': [],
            'changes_made': [],
            'errors': []
        }
        
        for repo in repos:
            try:
                repo_results = self._process_repository(repo)
                results['processed_repos'].append(repo_results)
                
                if repo_results['changes_made']:
                    results['changes_made'].extend(repo_results['changes_made'])
                
            except Exception as e:
                error_msg = f"Error processing {repo.name}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Step 8: Pin top repositories (run once for all repos)
        if not repo_name:  # Only when processing all repos
            try:
                sorted_repos = sorted(repos, key=lambda r: r.stargazers_count, reverse=True)
                top_repo_names = [r.full_name for r in sorted_repos[:6]]
                if self.github.pin_repositories(top_repo_names):
                    results['changes_made'].append("Pinned top 6 repositories by star count")
            except Exception as e:
                results['errors'].append(f"Failed to pin repositories: {str(e)}")
        
        return results
    
    def _process_repository(self, repo: RepoInfo) -> Dict[str, Any]:
        """Process a single repository through the hygiene checklist."""
        self.logger.info(f"Processing repository: {repo.name}")
        
        repo_changes = []
        
        # Step 1: Description, Website & Topics
        description = repo.description
        if not description or len(description.strip()) == 0:
            description = self.generate_description(repo.name)
            if len(description) > 120:
                description = description[:117] + "..."
        
        homepage = repo.homepage or f"https://{repo.owner}.github.io"
        
        topics = self.suggest_topics(repo)
        
        if self.github.update_repo_metadata(repo, description, homepage, topics):
            repo_changes.append(f"Updated metadata (description/homepage/topics)")
        
        # Step 6: Stale repository archive check
        if (repo.name == "Main-Project" and 
            datetime.now() - repo.last_pushed > timedelta(days=400)):
            if self.github.archive_repository(repo.full_name):
                repo_changes.append("Archived stale repository")
        
        return {
            'repo_name': repo.name,
            'changes_made': repo_changes
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Repository Hygiene Bot')
    parser.add_argument('--repo', help='Process specific repository only')
    parser.add_argument('--token', help='GitHub token (or use GITHUB_TOKEN env var)')
    args = parser.parse_args()
    
    try:
        bot = RepoHygieneBot(args.token)
        results = bot.run_hygiene_check(args.repo)
        
        print("\n=== Repository Hygiene Check Results ===")
        print(f"Processed {len(results['processed_repos'])} repositories")
        print(f"Made {len(results['changes_made'])} changes")
        
        if results['changes_made']:
            print("\nChanges made:")
            for change in results['changes_made']:
                print(f"  • {change}")
        
        if results['errors']:
            print(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  ! {error}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = f"hygiene_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())