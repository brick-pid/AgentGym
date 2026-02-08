"""
Test cases for SciWorld environment server.

This test suite covers:
- Basic API endpoints (health, create, reset, step, close)
- Error handling (invalid env_id, closed env, finished episode)
- Edge cases (multiple environments, sequential operations)

To run these tests:
1. Start the sciworld server: sciworld --host 0.0.0.0 --port 36001
2. Run: pytest tests/test_sciworld.py -v
"""

import pytest
import requests
import time

# Configuration
BASE_URL = "http://0.0.0.0:36001"
TIMEOUT = 30


class TestSciWorldBasic:
    """Test basic functionality of SciWorld server."""

    def test_health_check(self):
        """Test that the health endpoint returns ok status."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "sciworld"

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
        """Test resetting an environment with a task."""
        # Create environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]

        # Reset with task_id 0
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

        # Take a step
        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "look around"},
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


class TestSciWorldErrorHandling:
    """Test error handling in SciWorld server."""

    def test_step_invalid_env_id(self):
        """Test stepping with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": 99999, "action": "look around"},
            timeout=TIMEOUT
        )
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "ENV_NOT_FOUND"

    def test_reset_invalid_env_id(self):
        """Test resetting with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": 99999, "task_id": 0},
            timeout=TIMEOUT
        )
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "ENV_NOT_FOUND"

    def test_close_invalid_env_id(self):
        """Test closing with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/close",
            json={"env_id": 99999},
            timeout=TIMEOUT
        )
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "ENV_NOT_FOUND"

    def test_step_closed_environment(self):
        """Test stepping in a closed environment."""
        # Create and close environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

        # Try to step in closed environment
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "look around"},
            timeout=TIMEOUT
        )
        assert response.status_code == 409
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "ENV_CLOSED"

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
        assert response.status_code == 409
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "ENV_CLOSED"

    def test_step_before_reset(self):
        """Test stepping before resetting the environment."""
        # Create environment without reset
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]

        # Try to step without reset - should return 409 (episode not started)
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "look around"},
            timeout=TIMEOUT
        )
        assert response.status_code == 409
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "EPISODE_FINISHED"

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


class TestSciWorldWorkflow:
    """Test complete workflows in SciWorld server."""

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
        actions = ["look around", "inventory", "examine room"]
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
                json={"env_id": env_id, "action": "look around"},
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

    def test_reset_after_episode_done(self):
        """Test resetting an environment after an episode is finished."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Take steps until done or max steps
        max_steps = 50
        done = False
        for _ in range(max_steps):
            step_response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "look around"},
                timeout=TIMEOUT
            )
            if step_response.status_code == 200:
                data = step_response.json()
                if data.get("done", False):
                    done = True
                    break
            elif step_response.status_code == 409:
                # Episode finished
                done = True
                break

        # If episode is done, try to reset
        if done:
            reset_response = requests.post(
                f"{BASE_URL}/reset",
                json={"env_id": env_id, "task_id": 1},
                timeout=TIMEOUT
            )
            assert reset_response.status_code == 200

            # Should be able to step again after reset
            step_response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "look around"},
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
                json={"env_id": env_id, "action": "look around"},
                timeout=TIMEOUT
            )
            assert step_response.status_code == 200

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


class TestSciWorldActions:
    """Test various actions in SciWorld environment."""

    def test_common_actions(self):
        """Test common actions in the environment."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0},
            timeout=TIMEOUT
        )

        # Test various common actions
        common_actions = [
            "look around",
            "inventory",
            "examine room",
            "wait",
            "open door",
            "go north",
            "take object",
        ]

        for action in common_actions:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": action},
                timeout=TIMEOUT
            )
            # Action might be invalid, but should not crash
            assert response.status_code in [200, 400]
            if response.status_code == 200:
                data = response.json()
                assert "observation" in data
                assert "reward" in data
                assert "done" in data

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
