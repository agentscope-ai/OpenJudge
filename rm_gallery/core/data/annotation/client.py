"""
Label Studio API Client - programmatic interface for Label Studio project and annotation management.

Provides HTTP client functionality for interacting with Label Studio server including
project creation, task import, annotation export, and project management operations.
"""

from typing import Any, Dict, List, Optional

import requests
from loguru import logger


class LabelStudioClient:
    """
    HTTP client for Label Studio API providing programmatic access to annotation projects.

    Handles authentication, request formatting, and response processing for all
    Label Studio operations including project management and annotation workflows.

    Attributes:
        server_url: Base URL of the Label Studio server
        api_token: Authentication token for API access
        headers: HTTP headers including authorization and content type
    """

    def __init__(self, server_url: str, api_token: str):
        """
        Initialize Label Studio client with server connection details.

        Args:
            server_url: Base URL of Label Studio server (e.g., http://localhost:8080)
            api_token: API authentication token from Label Studio user account
        """
        self.server_url = server_url
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json",
        }

    def create_project(
        self, title: str, label_config: str, description: str = ""
    ) -> Optional[int]:
        """
        Create a new annotation project in Label Studio.

        Sets up a new project with specified labeling configuration and metadata
        for data annotation workflows.

        Args:
            title: Human-readable project title
            label_config: Label Studio XML configuration defining annotation interface
            description: Optional project description for documentation

        Returns:
            Project ID if creation successful, None if failed

        Raises:
            requests.RequestException: If HTTP request fails
        """
        try:
            project_data = {
                "title": title,
                "description": description,
                "label_config": label_config,
            }

            response = requests.post(
                f"{self.server_url}/api/projects/",
                json=project_data,
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 201:
                project = response.json()
                logger.info(f"Created project: {title} (ID: {project['id']})")
                return project["id"]
            else:
                logger.error(
                    f"Failed to create project: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return None

    def import_tasks(self, project_id: int, tasks: List[Dict[str, Any]]) -> bool:
        """
        Import annotation tasks to an existing project.

        Uploads data samples as tasks to be annotated within the specified project.
        Each task represents one item to be labeled by annotators.

        Args:
            project_id: Target project identifier
            tasks: List of task dictionaries containing data and metadata

        Returns:
            True if import successful, False otherwise

        Raises:
            requests.RequestException: If HTTP request fails
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/projects/{project_id}/import",
                json=tasks,
                headers=self.headers,
                timeout=60,
            )

            if response.status_code == 201:
                logger.info(
                    f"Successfully imported {len(tasks)} tasks to project {project_id}"
                )
                return True
            else:
                logger.error(
                    f"Failed to import tasks: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error importing tasks: {e}")
            return False

    def export_annotations(
        self, project_id: int, export_type: str = "JSON"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Export completed annotations from a project.

        Retrieves all annotation data from the project for analysis and
        downstream processing in reward model training pipelines.

        Args:
            project_id: Source project identifier
            export_type: Export format (JSON, CSV, etc.)

        Returns:
            List of annotation dictionaries if export successful, None otherwise

        Raises:
            requests.RequestException: If HTTP request fails
        """
        try:
            response = requests.get(
                f"{self.server_url}/api/projects/{project_id}/export",
                params={"exportType": export_type},
                headers=self.headers,
                timeout=60,
            )

            if response.status_code == 200:
                annotations = response.json()
                logger.info(
                    f"Exported {len(annotations)} annotations from project {project_id}"
                )
                return annotations
            else:
                logger.error(f"Failed to export annotations: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error exporting annotations: {e}")
            return None

    def delete_project(self, project_id: int) -> bool:
        """
        Delete an annotation project and all associated data.

        Permanently removes project, tasks, and annotations. Use with caution
        as this operation cannot be undone.

        Args:
            project_id: Project identifier to delete

        Returns:
            True if deletion successful, False otherwise

        Raises:
            requests.RequestException: If HTTP request fails
        """
        try:
            response = requests.delete(
                f"{self.server_url}/api/projects/{project_id}/",
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 204:
                logger.info(f"Successfully deleted project {project_id}")
                return True
            else:
                logger.error(f"Failed to delete project: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error deleting project: {e}")
            return False

    def get_projects(self) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve list of all accessible projects.

        Returns metadata for all projects accessible to the current user
        for project discovery and management operations.

        Returns:
            List of project dictionaries with metadata if successful, None otherwise

        Raises:
            requests.RequestException: If HTTP request fails
        """
        try:
            response = requests.get(
                f"{self.server_url}/api/projects/",
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 200:
                projects = response.json()
                logger.info(f"Retrieved {len(projects)} projects")
                return projects
            else:
                logger.error(f"Failed to get projects: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error getting projects: {e}")
            return None
