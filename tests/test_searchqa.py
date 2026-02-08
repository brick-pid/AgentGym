"""
Test cases for SearchQA environment server.

To run these tests:
1. Start the searchqa server: searchqa --host 0.0.0.0 --port 36004
2. Run: pytest tests/test_searchqa.py -v
"""

import pytest
import requests

# Configuration
BASE_URL = "http://127.0.0.1:36004"
TIMEOUT = 30


class TestSearchQABasic:
    """Test basic functionality of SearchQA server."""

    def test_health_check(self):
        """Test that the health endpoint returns ok status."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "searchqa"

    def test_create_environment(self):
        """Test creating a new environment."""
        response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "env_id" in data
        assert isinstance(data["env_id"], int)
        assert data["env_id"] >= 0

        # Clean up
        env_id = data["env_id"]
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_reset_environment(self):
        """Test resetting an environment."""
        # Create environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]

        # Reset with task_id
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )
        assert reset_response.status_code == 200
        data = reset_response.json()

        # Check response structure
        assert "observation" in data
        assert "info" in data
        assert isinstance(data["observation"], str)
        assert isinstance(data["info"], dict)
        # SearchQA observation should contain the question prompt
        assert "Question:" in data["observation"]

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_step_search_action(self):
        """Test taking a search step in the environment."""
        # Create and reset environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Take a search step
        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<search>test query</search>"},
            timeout=TIMEOUT
        )
        assert step_response.status_code == 200
        data = step_response.json()

        # Check response structure
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        assert isinstance(data["observation"], str)
        assert isinstance(data["reward"], (int, float))
        assert isinstance(data["done"], bool)
        # Search should return information
        assert "<information>" in data["observation"]

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_step_answer_action(self):
        """Test taking an answer step in the environment."""
        # Create and reset environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Take an answer step (likely incorrect)
        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<answer>test answer</answer>"},
            timeout=TIMEOUT
        )
        assert step_response.status_code == 200
        data = step_response.json()

        # Check response structure
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert isinstance(data["reward"], (int, float))
        assert isinstance(data["done"], bool)

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_close_environment(self):
        """Test closing an environment."""
        # Create environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]

        # Close environment
        close_response = requests.post(
            f"{BASE_URL}/close",
            json={"env_id": env_id},
            timeout=TIMEOUT
        )
        assert close_response.status_code == 200
        data = close_response.json()
        assert data["closed"] is True
        assert data["env_id"] == env_id


class TestSearchQAErrorHandling:
    """Test error handling in SearchQA server."""

    def test_step_invalid_env_id(self):
        """Test stepping with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": 99999, "action": "<search>test</search>"},
            timeout=TIMEOUT
        )
        # Should return error (IndexError results in 500)
        assert response.status_code in [404, 500]

    def test_reset_invalid_env_id(self):
        """Test resetting with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": 99999, "task_id": 0},
            timeout=TIMEOUT
        )
        # Should return error (IndexError results in 500)
        assert response.status_code in [404, 500]

    def test_close_invalid_env_id(self):
        """Test closing with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/close",
            json={"env_id": 99999},
            timeout=TIMEOUT
        )
        # SearchQA's close returns False for invalid env_id
        assert response.status_code == 200
        data = response.json()
        assert data["closed"] is False

    def test_step_closed_environment(self):
        """Test stepping in a closed environment."""
        # Create and close environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

        # Try to step in closed environment
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<search>test</search>"},
            timeout=TIMEOUT
        )
        # Should return error (IndexError results in 500)
        assert response.status_code in [404, 500]

    def test_step_invalid_action_format(self):
        """Test stepping with invalid action format."""
        # Create and reset environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Try invalid action format
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "invalid action without tags"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # Should return error message in observation
        assert "invalid" in data["observation"].lower()

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_reset_invalid_task_id(self):
        """Test resetting with an invalid task ID."""
        # Create environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]

        # Try invalid task_id (out of range)
        response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 999999},
            timeout=TIMEOUT
        )
        # Should return error (ValueError results in 500)
        assert response.status_code in [400, 500]

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


class TestSearchQAWorkflow:
    """Test complete workflows in SearchQA server."""

    def test_complete_search_workflow(self):
        """Test a complete search workflow: create -> reset -> search -> answer -> close."""
        # Create environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        assert create_response.status_code == 200
        env_id = create_response.json()["env_id"]

        # Reset environment
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )
        assert reset_response.status_code == 200

        # Perform search
        search_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<search>capital of France</search>"},
            timeout=TIMEOUT
        )
        assert search_response.status_code == 200
        assert "<information>" in search_response.json()["observation"]

        # Provide answer
        answer_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<answer>Paris</answer>"},
            timeout=TIMEOUT
        )
        assert answer_response.status_code == 200

        # Close environment
        close_response = requests.post(
            f"{BASE_URL}/close",
            json={"env_id": env_id},
            timeout=TIMEOUT
        )
        assert close_response.status_code == 200

    def test_multiple_environments(self):
        """Test creating and managing multiple environments simultaneously."""
        env_ids = []

        # Create multiple environments
        for i in range(3):
            response = requests.post(
                f"{BASE_URL}/create",
                json={"task_id": i},
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            env_ids.append(response.json()["env_id"])

        # Verify all env_ids are unique
        assert len(env_ids) == len(set(env_ids))

        # Reset all environments
        for env_id in env_ids:
            response = requests.post(
                f"{BASE_URL}/reset",
                json={"env_id": env_id, "task_id": 0},
                timeout=TIMEOUT
            )
            assert response.status_code == 200

        # Take steps in all environments
        for env_id in env_ids:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "<search>test</search>"},
                timeout=TIMEOUT
            )
            assert response.status_code == 200

        # Close all environments
        for env_id in env_ids:
            response = requests.post(
                f"{BASE_URL}/close",
                json={"env_id": env_id},
                timeout=TIMEOUT
            )
            assert response.status_code == 200

    def test_multiple_searches(self):
        """Test performing multiple searches in sequence."""
        # Create and reset environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Perform multiple searches
        search_queries = ["test query 1", "test query 2", "test query 3"]
        for query in search_queries:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": f"<search>{query}</search>"},
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            data = response.json()
            assert "<information>" in data["observation"]

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_reset_after_steps(self):
        """Test resetting an environment after taking some steps."""
        # Create and reset environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Take some steps
        requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<search>test</search>"},
            timeout=TIMEOUT
        )
        requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<answer>test</answer>"},
            timeout=TIMEOUT
        )

        # Reset again with different task
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 1},
            timeout=TIMEOUT
        )
        assert reset_response.status_code == 200

        # Should be able to step again after reset
        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<search>new query</search>"},
            timeout=TIMEOUT
        )
        assert step_response.status_code == 200

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


class TestSearchQAActions:
    """Test various actions in SearchQA environment."""

    def test_search_with_different_queries(self):
        """Test search action with different query types."""
        # Create and reset environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Test different search queries
        queries = [
            "simple query",
            "what is the capital of France?",
            "complex multi-word query with punctuation!",
        ]

        for query in queries:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": f"<search>{query}</search>"},
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            data = response.json()
            assert "<information>" in data["observation"]

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_empty_search_query(self):
        """Test search with empty query."""
        # Create and reset environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Try empty search
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "<search></search>"},
            timeout=TIMEOUT
        )
        # Should handle gracefully
        assert response.status_code in [200, 400]

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_answer_format_variations(self):
        """Test answer action with different formats."""
        # Create and reset environment
        create_response = requests.post(
            f"{BASE_URL}/create",
            json={"task_id": 0},
            timeout=TIMEOUT
        )
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Test different answer formats
        answers = [
            "<answer>short</answer>",
            "<answer>A longer answer with multiple words</answer>",
            "<answer>123</answer>",
        ]

        for answer in answers:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": answer},
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            data = response.json()
            assert "observation" in data
            assert "reward" in data

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
