"""
Test cases for WebShop environment server.

To run these tests:
1. Start the webshop server: webshop --host 0.0.0.0 --port 36003
2. Run: pytest tests/test_webshop.py -v
"""

import pytest
import requests

# Configuration
BASE_URL = "http://127.0.0.1:36003"
TIMEOUT = 30


class TestWebShopBasic:
    """Test basic functionality of WebShop server."""

    def test_health_check(self):
        """Test that the health endpoint returns ok status."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "webshop"

    def test_create_environment(self):
        """Test creating a new environment."""
        response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
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
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
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

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_reset_without_task_id(self):
        """Test resetting without specifying task_id."""
        # Create environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]

        # Reset without task_id (should use random session)
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id},
            timeout=TIMEOUT
        )
        assert reset_response.status_code == 200
        data = reset_response.json()
        assert "observation" in data
        assert isinstance(data["observation"], str)

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_step_environment(self):
        """Test taking a step in the environment."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Take a step with search action
        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "search[laptop]"},
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
        assert isinstance(data["info"], dict)

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_close_environment(self):
        """Test closing an environment."""
        # Create environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
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


class TestWebShopErrorHandling:
    """Test error handling in WebShop server."""

    def test_step_invalid_env_id(self):
        """Test stepping with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": 99999, "action": "search[test]"},
            timeout=TIMEOUT
        )
        # Should return error (KeyError results in 500)
        assert response.status_code in [404, 500]

    def test_reset_invalid_env_id(self):
        """Test resetting with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": 99999, "task_id": 0},
            timeout=TIMEOUT
        )
        # Should return error (KeyError results in 500)
        assert response.status_code in [404, 500]

    def test_close_invalid_env_id(self):
        """Test closing with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/close",
            json={"env_id": 99999},
            timeout=TIMEOUT
        )
        # WebShop raises IndexError for invalid env_id
        assert response.status_code in [404, 500]

    def test_step_closed_environment(self):
        """Test stepping in a closed environment."""
        # Create and close environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

        # Try to step in closed environment
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "search[test]"},
            timeout=TIMEOUT
        )
        # Should return error (KeyError results in 500)
        assert response.status_code in [404, 409, 500]

    def test_close_already_closed_environment(self):
        """Test closing an already closed environment."""
        # Create and close environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

        # Try to close again
        response = requests.post(
            f"{BASE_URL}/close",
            json={"env_id": env_id},
            timeout=TIMEOUT
        )
        # WebShop raises IndexError for already closed env
        assert response.status_code in [404, 409, 500]

    def test_step_before_reset(self):
        """Test stepping before resetting the environment."""
        # Create environment without reset
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]

        # WebShop auto-resets on create, so step should work
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "search[test]"},
            timeout=TIMEOUT
        )
        # Should work because webshop auto-resets
        assert response.status_code == 200

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


class TestWebShopWorkflow:
    """Test complete workflows in WebShop server."""

    def test_complete_episode_workflow(self):
        """Test a complete episode: create -> reset -> step -> close."""
        # Create environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        assert create_response.status_code == 200
        env_id = create_response.json()["env_id"]

        # Reset environment
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )
        assert reset_response.status_code == 200
        reset_data = reset_response.json()
        assert "observation" in reset_data

        # Take multiple steps
        actions = ["search[laptop]", "click[back to search]", "search[phone]"]
        for action in actions:
            step_response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": action},
                timeout=TIMEOUT
            )
            assert step_response.status_code == 200
            step_data = step_response.json()
            assert "observation" in step_data
            assert "done" in step_data

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
            response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
            assert response.status_code == 200
            env_ids.append(response.json()["env_id"])

        # Verify all env_ids are unique
        assert len(env_ids) == len(set(env_ids))

        # Reset all environments with different tasks
        for i, env_id in enumerate(env_ids):
            response = requests.post(
                f"{BASE_URL}/reset",
                json={"env_id": env_id, "task_id": i},
                timeout=TIMEOUT
            )
            assert response.status_code == 200

        # Take steps in all environments
        for env_id in env_ids:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "search[test]"},
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

    def test_reset_after_steps(self):
        """Test resetting an environment after taking some steps."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Take some steps
        for _ in range(3):
            requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "search[test]"},
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
            json={"env_id": env_id, "action": "search[laptop]"},
            timeout=TIMEOUT
        )
        assert step_response.status_code == 200

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_different_task_ids(self):
        """Test resetting with different task IDs."""
        # Create environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]

        # Test multiple task IDs
        task_ids = [0, 1, 2, 5, 10]
        for task_id in task_ids:
            reset_response = requests.post(
                f"{BASE_URL}/reset",
                json={"env_id": env_id, "task_id": task_id},
                timeout=TIMEOUT
            )
            assert reset_response.status_code == 200
            data = reset_response.json()
            assert "observation" in data
            assert "info" in data

            # Verify we can take a step after each reset
            step_response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "search[test]"},
                timeout=TIMEOUT
            )
            assert step_response.status_code == 200

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


class TestWebShopActions:
    """Test various actions in WebShop environment."""

    def test_search_action(self):
        """Test search action."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Test search action
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "search[laptop computer]"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert isinstance(data["observation"], str)

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_click_action(self):
        """Test click action."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Search first
        requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "search[laptop]"},
            timeout=TIMEOUT
        )

        # Test click action
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "click[back to search]"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_empty_action(self):
        """Test stepping with an empty action."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Try empty action
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": ""},
            timeout=TIMEOUT
        )
        # Should handle gracefully
        assert response.status_code in [200, 400]

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_sequential_actions(self):
        """Test a sequence of actions in a shopping workflow."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Simulate a shopping workflow
        actions = [
            "search[laptop]",
            "click[back to search]",
            "search[phone]",
            "click[back to search]",
        ]

        for action in actions:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": action},
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            data = response.json()
            assert "observation" in data
            assert "reward" in data
            assert "done" in data

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
