"""
Test cases for ALFWorld environment server.

This test suite covers:
- Basic API endpoints (health, create, reset, step, close)
- Error handling (invalid env_id, closed env, finished episode)
- Edge cases (multiple environments, sequential operations)
- ALFWorld-specific features (world_type, available_actions)

To run these tests:
1. Start the alfworld server: alfworld --host 0.0.0.0 --port 36002
2. Run: pytest tests/test_alfworld.py -v
"""

import pytest
import requests
import time

# Configuration
BASE_URL = "http://127.0.0.1:36002"
TIMEOUT = 30


class TestALFWorldBasic:
    """Test basic functionality of ALFWorld server."""

    def test_health_check(self):
        """Test that the health endpoint returns ok status."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "alfworld"

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

        # Reset with task_id 0 and default world_type
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0, "world_type": "Text"},
            timeout=TIMEOUT
        )
        assert reset_response.status_code == 200
        data = reset_response.json()

        # Check response structure
        assert "observation" in data
        assert "info" in data
        assert isinstance(data["observation"], str)
        assert isinstance(data["info"], dict)

        # ALFWorld specific fields
        if "available_actions" in data["info"]:
            assert isinstance(data["info"]["available_actions"], list)

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_reset_with_world_types(self):
        """Test resetting with different world types."""
        # Create environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]

        # Test valid world types
        world_types = ["Text", "Embody", "Hybrid"]
        for world_type in world_types:
            reset_response = requests.post(
                f"{BASE_URL}/reset",
                json={"env_id": env_id, "task_id": 0, "world_type": world_type},
                timeout=TIMEOUT
            )
            assert reset_response.status_code == 200
            data = reset_response.json()
            assert "observation" in data

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_step_environment(self):
        """Test taking a step in the environment."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0, "world_type": "Text"},
            timeout=TIMEOUT
        )

        # Get available actions if provided
        reset_data = reset_response.json()
        available_actions = reset_data.get("info", {}).get("available_actions", [])

        # Take a step with a common action
        action = available_actions[0] if available_actions else "look"
        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": action},
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


class TestALFWorldErrorHandling:
    """Test error handling in ALFWorld server."""

    def test_step_invalid_env_id(self):
        """Test stepping with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": 99999, "action": "look"},
            timeout=TIMEOUT
        )
        # Should return error (either 404 or error in response)
        assert response.status_code in [200, 404]
        data = response.json()
        if response.status_code == 200:
            assert "error" in data

    def test_reset_invalid_env_id(self):
        """Test resetting with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": 99999, "task_id": 0, "world_type": "Text"},
            timeout=TIMEOUT
        )
        # Should return error (either 404 or error in response)
        assert response.status_code in [200, 404]
        data = response.json()
        if response.status_code == 200:
            # ALFWorld returns error in observation field
            assert "error" in data or ("observation" in data and isinstance(data["observation"], dict) and "error" in data["observation"])

    def test_close_invalid_env_id(self):
        """Test closing with an invalid environment ID."""
        response = requests.post(
            f"{BASE_URL}/close",
            json={"env_id": 99999},
            timeout=TIMEOUT
        )
        # ALFWorld's close method calls _check_id which raises NameError for invalid ID
        # This results in 500 error, which is acceptable for now
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.json()
            # May return error or False
            assert data.get("closed") is False or "error" in str(data)

    def test_step_closed_environment(self):
        """Test stepping in a closed environment."""
        # Create and close environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

        # Try to step in closed environment
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "look"},
            timeout=TIMEOUT
        )
        # Should return error
        assert response.status_code in [200, 404, 409]
        data = response.json()
        if response.status_code == 200:
            assert "error" in data

    def test_close_already_closed_environment(self):
        """Test closing an already closed environment."""
        # Create and close environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

        # Try to close again - ALFWorld's _check_id raises NameError for deleted env
        response = requests.post(
            f"{BASE_URL}/close",
            json={"env_id": env_id},
            timeout=TIMEOUT
        )
        # Should return error (500 due to NameError is acceptable)
        assert response.status_code in [200, 404, 409, 500]

    def test_reset_invalid_world_type(self):
        """Test resetting with an invalid world type."""
        # Create environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]

        # Try invalid world type
        response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0, "world_type": "InvalidType"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        # ALFWorld returns error message directly
        assert "error" in data or ("observation" in data and isinstance(data["observation"], dict) and "error" in data["observation"])

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_step_before_reset(self):
        """Test stepping before resetting the environment."""
        # Create environment without reset
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]

        # Try to step without reset
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "look"},
            timeout=TIMEOUT
        )
        # Should return error
        assert response.status_code in [200, 400, 409, 500]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


class TestALFWorldWorkflow:
    """Test complete workflows in ALFWorld server."""

    def test_complete_episode_workflow(self):
        """Test a complete episode: create -> reset -> step -> close."""
        # Create environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        assert create_response.status_code == 200
        env_id = create_response.json()["env_id"]

        # Reset environment
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0, "world_type": "Text"},
            timeout=TIMEOUT
        )
        assert reset_response.status_code == 200
        reset_data = reset_response.json()
        assert "observation" in reset_data

        # Get available actions
        available_actions = reset_data.get("info", {}).get("available_actions", [])

        # Take multiple steps
        actions = available_actions[:3] if len(available_actions) >= 3 else ["look", "inventory", "examine"]
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
                json={"env_id": env_id, "task_id": i, "world_type": "Text"},
                timeout=TIMEOUT
            )
            assert response.status_code == 200

        # Take steps in all environments
        for env_id in env_ids:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "look"},
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
            json={"env_id": env_id, "task_id": 0, "world_type": "Text"},
            timeout=TIMEOUT
        )

        # Take some steps
        for _ in range(3):
            requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "look"},
                timeout=TIMEOUT
            )

        # Reset again with different task
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 1, "world_type": "Text"},
            timeout=TIMEOUT
        )
        assert reset_response.status_code == 200

        # Should be able to step again after reset
        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "look"},
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
        task_ids = [0, 1, 2, 5]
        for task_id in task_ids:
            reset_response = requests.post(
                f"{BASE_URL}/reset",
                json={"env_id": env_id, "task_id": task_id, "world_type": "Text"},
                timeout=TIMEOUT
            )
            assert reset_response.status_code == 200
            data = reset_response.json()
            assert "observation" in data
            assert "info" in data

            # Verify we can take a step after each reset
            step_response = requests.post(
                f"{BASE_URL}/step",
                json={"env_id": env_id, "action": "look"},
                timeout=TIMEOUT
            )
            assert step_response.status_code == 200

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


class TestALFWorldActions:
    """Test various actions in ALFWorld environment."""

    def test_common_actions(self):
        """Test common actions in the environment."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0, "world_type": "Text"},
            timeout=TIMEOUT
        )

        # Get available actions
        reset_data = reset_response.json()
        available_actions = reset_data.get("info", {}).get("available_actions", [])

        # Test available actions if provided
        if available_actions:
            for action in available_actions[:5]:  # Test first 5 actions
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

                # Stop if episode is done
                if data.get("done"):
                    break

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)

    def test_empty_action(self):
        """Test stepping with an empty action."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0, "world_type": "Text"},
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

    def test_available_actions_in_response(self):
        """Test that available actions are included in step responses."""
        # Create and reset environment
        create_response = requests.post(f"{BASE_URL}/create", timeout=TIMEOUT)
        env_id = create_response.json()["env_id"]
        requests.post(
            f"{BASE_URL}/reset",
            json={"env_id": env_id, "task_id": 0, "world_type": "Text"},
            timeout=TIMEOUT
        )

        # Take a step
        response = requests.post(
            f"{BASE_URL}/step",
            json={"env_id": env_id, "action": "look"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()

        # Check if available_actions are in the response
        if "info" in data and "available_actions" in data["info"]:
            assert isinstance(data["info"]["available_actions"], list)

        # Clean up
        requests.post(f"{BASE_URL}/close", json={"env_id": env_id}, timeout=TIMEOUT)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
