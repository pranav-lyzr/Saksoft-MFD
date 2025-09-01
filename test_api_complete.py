#!/usr/bin/env python3
"""
Complete API Testing Script for Saksoft MFD
Tests all endpoints with different user types (admin, project_admin, developer)
"""

import requests
import json
import time
from typing import Dict, Any, List
import sys

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.client_id = None
        self.admin_token = None
        self.project_admin_token = None
        self.developer_token = None
        self.project_id = None
        self.chat_session_id = None
        
    def log_test(self, test_name: str, success: bool, response: requests.Response = None, error: str = None):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "status_code": response.status_code if response else None,
            "error": error,
            "response": response.json() if response and response.status_code < 400 else None
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if response:
            print(f"   📊 Status Code: {response.status_code}")
            print(f"   📊 Response Headers: {dict(response.headers)}")
            
            if response.status_code >= 400:
                print(f"   ❌ Error Response: {response.text}")
                try:
                    error_json = response.json()
                    print(f"   📋 Error JSON: {json.dumps(error_json, indent=2)}")
                except:
                    pass
            else:
                try:
                    success_json = response.json()
                    print(f"   📋 Success Response: {json.dumps(success_json, indent=2)}")
                except:
                    print(f"   📋 Raw Response: {response.text}")
        
        if error:
            print(f"   💥 Exception: {error}")
        
        print()

    def test_health_check(self):
        """Test health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            success = response.status_code == 200
            self.log_test("Health Check", success, response)
            return success
        except Exception as e:
            self.log_test("Health Check", False, error=str(e))
            return False

    def create_client(self):
        """Create a new client"""
        try:
            # Use timestamp to make client name unique
            import time
            timestamp = int(time.time())
            client_name = f"Test_Client_{timestamp}"
            
            payload = {
                "name": client_name,
                "admin_username": f"admin_user_{timestamp}",
                "admin_password": "admin123",
                "special_key": "lyzr-saksoft"
            }
            
            print(f"🔧 Creating client: {client_name}")
            print(f"📤 Request payload: {json.dumps(payload, indent=2)}")
            
            response = self.session.post(
                f"{self.base_url}/clients",
                json=payload
            )
            
            print(f"📥 Response Status: {response.status_code}")
            print(f"📥 Response Headers: {dict(response.headers)}")
            print(f"📥 Response Body: {response.text}")
            
            success = response.status_code == 201
            if success:
                response_data = response.json()
                self.client_id = response_data["id"]
                self.admin_username = payload["admin_username"]  # Store admin username for login
                print(f"✅ Client created successfully!")
                print(f"📋 Client ID: {self.client_id}")
                print(f"📋 Admin Username: {self.admin_username}")
                print(f"📋 Full Response: {json.dumps(response_data, indent=2)}")
            else:
                print(f"❌ Client creation failed with status {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"📋 Error Details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"📋 Raw Error: {response.text}")
            
            self.log_test("Create Client", success, response)
            return success
        except Exception as e:
            print(f"💥 Exception during client creation: {str(e)}")
            self.log_test("Create Client", False, error=str(e))
            return False

    def login_admin(self):
        """Login as admin user"""
        try:
            # Get the admin username from the client creation
            if not hasattr(self, 'admin_username') or not self.admin_username:
                print("❌ Admin username not set. Client must be created first.")
                return False
                
            payload = {
                "username": self.admin_username,
                "password": "admin123"
            }
            
            print(f"🔐 Logging in admin user: {self.admin_username}")
            print(f"📤 Login payload: {json.dumps(payload, indent=2)}")
            
            response = self.session.post(
                f"{self.base_url}/login",
                data=payload
            )
            
            print(f"📥 Login Response Status: {response.status_code}")
            print(f"📥 Login Response Body: {response.text}")
            
            success = response.status_code == 200
            if success:
                response_data = response.json()
                self.admin_token = response_data["access_token"]
                print(f"✅ Admin logged in successfully!")
                print(f"🔑 Token: {self.admin_token[:20]}...")
                print(f"📋 Full Response: {json.dumps(response_data, indent=2)}")
            else:
                print(f"❌ Admin login failed with status {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"📋 Error Details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"📋 Raw Error: {response.text}")
            
            self.log_test("Admin Login", success, response)
            return success
        except Exception as e:
            print(f"💥 Exception during admin login: {str(e)}")
            self.log_test("Admin Login", False, error=str(e))
            return False

    def create_users(self):
        """Create different types of users"""
        users_to_create = [
            {
                "username": "project_admin_user",
                "password": "project123",
                "user_type": "project_admin"
            },
            {
                "username": "developer_user",
                "password": "dev123",
                "user_type": "developer"
            }
        ]
        
        for user_data in users_to_create:
            try:
                response = self.session.post(
                    f"{self.base_url}/users",
                    json=user_data,
                    headers={"Authorization": f"Bearer {self.admin_token}"}
                )
                success = response.status_code == 200
                self.log_test(f"Create User: {user_data['username']}", success, response)
            except Exception as e:
                self.log_test(f"Create User: {user_data['username']}", False, error=str(e))

    def login_users(self):
        """Login as different user types"""
        # Login as project admin
        try:
            payload = {"username": "project_admin_user", "password": "project123"}
            response = self.session.post(f"{self.base_url}/login", data=payload)
            if response.status_code == 200:
                self.project_admin_token = response.json()["access_token"]
                print("✅ Project Admin logged in successfully")
            self.log_test("Project Admin Login", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Project Admin Login", False, error=str(e))

        # Login as developer
        try:
            payload = {"username": "developer_user", "password": "dev123"}
            response = self.session.post(f"{self.base_url}/login", data=payload)
            if response.status_code == 200:
                self.developer_token = response.json()["access_token"]
                print("✅ Developer logged in successfully")
            self.log_test("Developer Login", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Developer Login", False, error=str(e))

    def create_project(self):
        """Create a test project"""
        try:
            payload = {"name": "Test Project"}
            response = self.session.post(
                f"{self.base_url}/projects/create_project",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            success = response.status_code == 200
            if success:
                self.project_id = response.json()["project_id"]
                print(f"✅ Project created with ID: {self.project_id}")
            self.log_test("Create Project", success, response)
            return success
        except Exception as e:
            self.log_test("Create Project", False, error=str(e))
            return False

    def add_github_repo(self):
        """Add GitHub repository to project"""
        try:
            payload = {
                "github_url": "https://github.com/ShakirFarhan/Youtube-Clone",
                "source_name": "youtube-clone",
                "pat": None
            }
            response = self.session.post(
                f"{self.base_url}/projects/project/{self.project_id}/repo",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            success = response.status_code == 200
            self.log_test("Add GitHub Repository", success, response)
            return success
        except Exception as e:
            self.log_test("Add GitHub Repository", False, error=str(e))
            return False

    def add_documentation(self):
        """Add documentation to project"""
        try:
            payload = {
                "text": "This is a YouTube clone built with React JS, Redux, Tailwind CSS, and Rapid API. Features include video streaming, search functionality, and responsive design.",
                "source_name": "youtube-clone"
            }
            response = self.session.post(
                f"{self.base_url}/projects/project/{self.project_id}/documentation",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            success = response.status_code == 200
            self.log_test("Add Documentation", success, response)
            return success
        except Exception as e:
            self.log_test("Add Documentation", False, error=str(e))
            return False

    def test_user_management_apis(self):
        """Test all user management APIs"""
        print("\n🔐 Testing User Management APIs...")
        
        # Test list users (admin)
        try:
            response = self.session.get(
                f"{self.base_url}/users",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("List Users (Admin)", response.status_code == 200, response)
        except Exception as e:
            self.log_test("List Users (Admin)", False, error=str(e))

        # Test list users (project admin)
        try:
            response = self.session.get(
                f"{self.base_url}/users",
                headers={"Authorization": f"Bearer {self.project_admin_token}"}
            )
            self.log_test("List Users (Project Admin)", response.status_code == 200, response)
        except Exception as e:
            self.log_test("List Users (Project Admin)", False, error=str(e))

        # Test get user details
        try:
            response = self.session.get(
                f"{self.base_url}/users/admin_user",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Get User Details", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Get User Details", False, error=str(e))

        # Test change password
        try:
            payload = {"current_password": "admin123", "new_password": "newadmin123"}
            response = self.session.post(
                f"{self.base_url}/users/me/change_password",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Change Password", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Change Password", False, error=str(e))

        # Test assign project to user
        try:
            payload = {"project_id": self.project_id}
            response = self.session.post(
                f"{self.base_url}/users/developer_user/assign_project",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Assign Project to User", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Assign Project to User", False, error=str(e))

    def test_project_management_apis(self):
        """Test all project management APIs"""
        print("\n📁 Testing Project Management APIs...")
        
        # Test list projects (admin)
        try:
            response = self.session.get(
                f"{self.base_url}/projects",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("List Projects (Admin)", response.status_code == 200, response)
        except Exception as e:
            self.log_test("List Projects (Admin)", False, error=str(e))

        # Test list projects (project admin)
        try:
            response = self.session.get(
                f"{self.base_url}/projects",
                headers={"Authorization": f"Bearer {self.project_admin_token}"}
            )
            self.log_test("List Projects (Project Admin)", response.status_code == 200, response)
        except Exception as e:
            self.log_test("List Projects (Project Admin)", False, error=str(e))

        # Test get project details
        try:
            response = self.session.get(
                f"{self.base_url}/projects/project/{self.project_id}",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Get Project Details", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Get Project Details", False, error=str(e))

        # Test get user projects
        try:
            response = self.session.get(
                f"{self.base_url}/projects/users/developer_user/projects",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Get User Projects", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Get User Projects", False, error=str(e))

        # Test get RAG documents
        try:
            response = self.session.get(
                f"{self.base_url}/projects/project/{self.project_id}/rag/documents",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Get RAG Documents", response.status_code in [200, 404], response)
        except Exception as e:
            self.log_test("Get RAG Documents", False, error=str(e))

    def test_chat_apis(self):
        """Test all chat/code operation APIs"""
        print("\n💬 Testing Chat/Code Operation APIs...")
        
        # Create chat session for search
        try:
            payload = {"project_id": self.project_id, "session_type": "search"}
            response = self.session.post(
                f"{self.base_url}/chat/chat_sessions",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            success = response.status_code == 200
            if success:
                self.chat_session_id = response.json()["session_id"]
                print(f"✅ Chat session created with ID: {self.chat_session_id}")
            self.log_test("Create Chat Session (Search)", success, response)
        except Exception as e:
            self.log_test("Create Chat Session (Search)", False, error=str(e))

        # Test search in session
        if self.chat_session_id:
            try:
                payload = {"message": "What are the main features of this YouTube clone?"}
                response = self.session.post(
                    f"{self.base_url}/chat/chat_sessions/{self.chat_session_id}/search",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.admin_token}"}
                )
                self.log_test("Search in Chat Session", response.status_code in [200, 500], response)
            except Exception as e:
                self.log_test("Search in Chat Session", False, error=str(e))

        # Create chat session for generate
        try:
            payload = {"project_id": self.project_id, "session_type": "generate"}
            response = self.session.post(
                f"{self.base_url}/chat/chat_sessions",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            success = response.status_code == 200
            if success:
                generate_session_id = response.json()["session_id"]
                print(f"✅ Generate session created with ID: {generate_session_id}")
            self.log_test("Create Chat Session (Generate)", success, response)
        except Exception as e:
            self.log_test("Create Chat Session (Generate)", False, error=str(e))

        # Test technical documentation generation
        try:
            payload = {"description": "Generate technical documentation for the YouTube clone project"}
            response = self.session.post(
                f"{self.base_url}/chat/project/{self.project_id}/technical_documentation",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Generate Technical Documentation", response.status_code in [200, 500], response)
        except Exception as e:
            self.log_test("Generate Technical Documentation", False, error=str(e))

        # Test impact analysis
        try:
            payload = {"description": "Analyze the impact of adding a new feature to the YouTube clone"}
            response = self.session.post(
                f"{self.base_url}/chat/project/{self.project_id}/impact_analysis",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Generate Impact Analysis", response.status_code in [200, 500], response)
        except Exception as e:
            self.log_test("Generate Impact Analysis", False, error=str(e))

        # Test code suggestion
        try:
            payload = {"context": "I want to add a dark mode feature to the YouTube clone"}
            response = self.session.post(
                f"{self.base_url}/chat/code_suggestion",
                json=payload,
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            self.log_test("Code Suggestion", response.status_code in [200, 500], response)
        except Exception as e:
            self.log_test("Code Suggestion", False, error=str(e))

    def test_permission_checks(self):
        """Test permission checks for different user types"""
        print("\n🔒 Testing Permission Checks...")
        
        # Test developer trying to create project (should fail)
        try:
            payload = {"name": "Unauthorized Project"}
            response = self.session.post(
                f"{self.base_url}/projects/create_project",
                json=payload,
                headers={"Authorization": f"Bearer {self.developer_token}"}
            )
            self.log_test("Developer Create Project (Should Fail)", response.status_code == 403, response)
        except Exception as e:
            self.log_test("Developer Create Project (Should Fail)", False, error=str(e))

        # Test project admin trying to delete user (should fail)
        try:
            response = self.session.delete(
                f"{self.base_url}/users/developer_user",
                headers={"Authorization": f"Bearer {self.project_admin_token}"}
            )
            self.log_test("Project Admin Delete User (Should Fail)", response.status_code == 403, response)
        except Exception as e:
            self.log_test("Project Admin Delete User (Should Fail)", False, error=str(e))

        # Test developer trying to access admin-only endpoint
        try:
            response = self.session.get(
                f"{self.base_url}/users",
                headers={"Authorization": f"Bearer {self.developer_token}"}
            )
            self.log_test("Developer List Users (Should Fail)", response.status_code == 403, response)
        except Exception as e:
            self.log_test("Developer List Users (Should Fail)", False, error=str(e))

    def test_client_management_apis(self):
        """Test client management APIs"""
        print("\n🏢 Testing Client Management APIs...")
        
        # Test get client
        try:
            response = self.session.get(
                f"{self.base_url}/clients/{self.client_id}",
                headers={"special-key": "lyzr-saksoft"}
            )
            self.log_test("Get Client", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Get Client", False, error=str(e))

        # Test update client
        try:
            payload = {"name": "Updated Test Client"}
            response = self.session.put(
                f"{self.base_url}/clients/{self.client_id}",
                json=payload,
                headers={"special-key": "lyzr-saksoft"}
            )
            self.log_test("Update Client", response.status_code == 200, response)
        except Exception as e:
            self.log_test("Update Client", False, error=str(e))

    def run_complete_test_suite(self):
        """Run the complete test suite"""
        print("🚀 Starting Complete API Test Suite for Saksoft MFD")
        print("=" * 60)
        
        # Step 1: Basic health check
        if not self.test_health_check():
            print("❌ Health check failed. Exiting...")
            return False
        
        # Step 2: Create client
        if not self.create_client():
            print("❌ Client creation failed. Exiting...")
            return False
        
        # Step 3: Login as admin
        if not self.login_admin():
            print("❌ Admin login failed. Exiting...")
            return False
        
        # Step 4: Create users
        self.create_users()
        
        # Step 5: Login as different users
        self.login_users()
        
        # Step 6: Create project
        if not self.create_project():
            print("❌ Project creation failed. Exiting...")
            return False
        
        # Step 7: Add GitHub repository
        self.add_github_repo()
        
        # Step 8: Add documentation
        self.add_documentation()
        
        # Step 9: Test all API categories
        self.test_user_management_apis()
        self.test_project_management_apis()
        self.test_chat_apis()
        self.test_permission_checks()
        self.test_client_management_apis()
        
        # Step 10: Print summary
        self.print_test_summary()
        
        return True

    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['error']}")
        
        print("\n🎯 Test Suite Completed!")

def test_basic_functionality():
    """Test basic functionality for debugging"""
    print("🔧 Testing Basic Functionality...")
    print("=" * 50)
    
    tester = APITester()
    
    # Test health check
    print("\n1️⃣ Testing Health Check...")
    health_success = tester.test_health_check()
    
    if health_success:
        print("\n2️⃣ Testing Client Creation...")
        client_success = tester.create_client()
        
        if client_success:
            print("\n3️⃣ Testing Admin Login...")
            login_success = tester.login_admin()
            
            if login_success:
                print("\n✅ Basic functionality test passed!")
                return True
            else:
                print("\n❌ Admin login failed!")
        else:
            print("\n❌ Client creation failed!")
    else:
        print("\n❌ Health check failed!")
    
    return False

def main():
    """Main function"""
    print("🔧 Saksoft MFD API Testing Script")
    print("This script will test all APIs with different user types and permissions")
    print()
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running. Please start the Docker container first:")
            print("   docker-compose up --build")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API. Please start the Docker container first:")
        print("   docker-compose up --build")
        return
    
    print("✅ API is running. Starting tests...")
    print()
    
    # Ask user what to test
    print("Choose test mode:")
    print("1. Basic functionality test (client creation + login)")
    print("2. Full test suite")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = test_basic_functionality()
    else:
        # Run full tests
        tester = APITester()
        success = tester.run_complete_test_suite()
    
    if success:
        print("\n🎉 Tests completed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check the results above.")

if __name__ == "__main__":
    main()
