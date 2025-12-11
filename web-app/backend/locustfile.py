"""
Locust load testing file for LLM Judge Auditor API

Usage:
    locust -f locustfile.py --host=http://localhost:8000

Or with web UI:
    locust -f locustfile.py --host=http://localhost:8000 --web-host=0.0.0.0 --web-port=8089
"""

from locust import HttpUser, task, between, events
import json
import random
import time
from datetime import datetime


class LLMJudgeAuditorUser(HttpUser):
    """Simulates a user interacting with the LLM Judge Auditor API"""
    
    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)
    
    def on_start(self):
        """Called when a simulated user starts"""
        self.register_and_login()
    
    def register_and_login(self):
        """Register a new user and login"""
        timestamp = int(time.time() * 1000)
        username = f"loadtest_user_{timestamp}_{random.randint(1000, 9999)}"
        email = f"{username}@example.com"
        password = "LoadTest123!"
        
        # Register
        response = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password
            },
            name="/api/v1/auth/register"
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.username = username
        else:
            # If registration fails, try login (user might already exist)
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": username,
                    "password": password
                },
                name="/api/v1/auth/login"
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.username = username
    
    def get_auth_headers(self):
        """Get authorization headers"""
        return {"Authorization": f"Bearer {self.token}"}
    
    @task(5)
    def health_check(self):
        """Check API health (most frequent task)"""
        self.client.get("/health", name="/health")
    
    @task(3)
    def detailed_health_check(self):
        """Check detailed health"""
        self.client.get("/health/detailed", name="/health/detailed")
    
    @task(10)
    def create_evaluation(self):
        """Create a new evaluation (core functionality)"""
        source_texts = [
            "The capital of France is Paris.",
            "Python is a programming language.",
            "The Earth orbits around the Sun.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Shakespeare wrote Romeo and Juliet."
        ]
        
        candidate_outputs = [
            "Paris is the capital city of France and one of the most visited cities in the world.",
            "Python is a high-level, interpreted programming language known for its simplicity.",
            "The Earth takes approximately 365.25 days to complete one orbit around the Sun.",
            "At standard atmospheric pressure, water boils at 100°C or 212°F.",
            "Romeo and Juliet is a tragedy written by William Shakespeare in the 1590s."
        ]
        
        idx = random.randint(0, len(source_texts) - 1)
        
        response = self.client.post(
            "/api/v1/evaluations",
            json={
                "source_text": source_texts[idx],
                "candidate_output": candidate_outputs[idx],
                "config": {
                    "judge_models": ["gpt-4"],
                    "enable_retrieval": False,
                    "aggregation_strategy": "mean"
                }
            },
            headers=self.get_auth_headers(),
            name="/api/v1/evaluations [POST]"
        )
        
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            if session_id:
                # Store for later retrieval
                if not hasattr(self, 'session_ids'):
                    self.session_ids = []
                self.session_ids.append(session_id)
    
    @task(7)
    def get_evaluation(self):
        """Retrieve an evaluation"""
        if hasattr(self, 'session_ids') and self.session_ids:
            session_id = random.choice(self.session_ids)
            self.client.get(
                f"/api/v1/evaluations/{session_id}",
                headers=self.get_auth_headers(),
                name="/api/v1/evaluations/{id} [GET]"
            )
    
    @task(5)
    def list_evaluations(self):
        """List evaluation history"""
        page = random.randint(1, 3)
        self.client.get(
            f"/api/v1/evaluations?page={page}&limit=10",
            headers=self.get_auth_headers(),
            name="/api/v1/evaluations [GET]"
        )
    
    @task(2)
    def get_preferences(self):
        """Get user preferences"""
        self.client.get(
            "/api/v1/preferences",
            headers=self.get_auth_headers(),
            name="/api/v1/preferences [GET]"
        )
    
    @task(1)
    def update_preferences(self):
        """Update user preferences"""
        judge_models = [
            ["gpt-4"],
            ["gpt-4", "claude-3"],
            ["gpt-4", "claude-3", "gemini-pro"]
        ]
        
        self.client.put(
            "/api/v1/preferences",
            json={
                "judge_models": random.choice(judge_models),
                "enable_retrieval": random.choice([True, False]),
                "aggregation_strategy": random.choice(["mean", "weighted_mean", "median"])
            },
            headers=self.get_auth_headers(),
            name="/api/v1/preferences [PUT]"
        )
    
    @task(1)
    def export_evaluation(self):
        """Export evaluation results"""
        if hasattr(self, 'session_ids') and self.session_ids:
            session_id = random.choice(self.session_ids)
            format_type = random.choice(["json", "csv"])
            self.client.get(
                f"/api/v1/evaluations/{session_id}/export?format={format_type}",
                headers=self.get_auth_headers(),
                name=f"/api/v1/evaluations/{{id}}/export?format={format_type}"
            )
    
    @task(1)
    def get_metrics(self):
        """Get application metrics"""
        self.client.get("/metrics", name="/metrics")


class QuickTestUser(HttpUser):
    """Quick test user for rapid testing"""
    
    wait_time = between(0.5, 2)
    
    @task
    def quick_health_check(self):
        """Rapid health checks"""
        self.client.get("/health")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print(f"\n{'='*60}")
    print(f"Load Test Started: {datetime.now().isoformat()}")
    print(f"Target Host: {environment.host}")
    print(f"{'='*60}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print(f"\n{'='*60}")
    print(f"Load Test Completed: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    # Print summary statistics
    stats = environment.stats
    print("\nRequest Statistics:")
    print(f"  Total Requests: {stats.total.num_requests}")
    print(f"  Total Failures: {stats.total.num_failures}")
    print(f"  Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"  Min Response Time: {stats.total.min_response_time:.2f}ms")
    print(f"  Max Response Time: {stats.total.max_response_time:.2f}ms")
    print(f"  Requests per Second: {stats.total.total_rps:.2f}")
    print(f"  Failure Rate: {(stats.total.num_failures / max(stats.total.num_requests, 1) * 100):.2f}%")
    print()


# Custom load shapes for different test scenarios
from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    A step load shape that increases users in steps
    
    Usage:
        locust -f locustfile.py --host=http://localhost:8000 --headless --users=100 --spawn-rate=10
    """
    
    step_time = 30  # seconds per step
    step_load = 10  # users per step
    spawn_rate = 5
    time_limit = 300  # 5 minutes total
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        return (current_step * self.step_load, self.spawn_rate)


class SpikeLoadShape(LoadTestShape):
    """
    A spike load shape that simulates traffic spikes
    """
    
    time_limit = 300
    spawn_rate = 10
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        # Create spikes every 60 seconds
        if run_time % 60 < 10:
            # Spike: 100 users
            return (100, self.spawn_rate)
        else:
            # Normal: 20 users
            return (20, self.spawn_rate)
