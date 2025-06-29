"""Authentication and authorization module for RAG Agent RBAC."""

import streamlit as st
import yaml
import bcrypt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles in the system."""
    ADMIN = "admin"
    USER = "user" 
    VIEWER = "viewer"

@dataclass
class User:
    """User data structure."""
    username: str
    name: str
    email: str
    role: UserRole
    hashed_password: str
    is_active: bool = True

class Permission(Enum):
    """System permissions."""
    VIEW_DOCUMENTS = "view_documents"
    SEARCH_DOCUMENTS = "search_documents"
    UPLOAD_DOCUMENTS = "upload_documents"
    DELETE_DOCUMENTS = "delete_documents"
    VIEW_SYSTEM_STATUS = "view_system_status"
    MANAGE_USERS = "manage_users"
    CONFIGURE_SYSTEM = "configure_system"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.VIEW_DOCUMENTS,
        Permission.SEARCH_DOCUMENTS,
        Permission.UPLOAD_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
        Permission.VIEW_SYSTEM_STATUS,
        Permission.MANAGE_USERS,
        Permission.CONFIGURE_SYSTEM,
    ],
    UserRole.USER: [
        Permission.VIEW_DOCUMENTS,
        Permission.SEARCH_DOCUMENTS,
        Permission.UPLOAD_DOCUMENTS,
        Permission.VIEW_SYSTEM_STATUS,
    ],
    UserRole.VIEWER: [
        Permission.VIEW_DOCUMENTS,
        Permission.SEARCH_DOCUMENTS,
    ],
}

class AuthManager:
    """Manages authentication and authorization."""
    
    def __init__(self, users_file: str = None):
        """Initialize auth manager with users file."""
        if users_file is None:
            # Default to users.yaml in the project root directory
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            users_file = project_root / "users.yaml"
        
        self.users_file = Path(users_file)
        self.users: Dict[str, User] = {}
        self._load_users()
        
        # Create default admin if no users exist
        if not self.users:
            self._create_default_admin()
    
    def _load_users(self) -> None:
        """Load users from YAML file."""
        if not self.users_file.exists():
            logger.info(f"Users file {self.users_file} not found, starting with empty user base")
            return
            
        try:
            with open(self.users_file, 'r') as f:
                data = yaml.safe_load(f) or {}
                
            for username, user_data in data.get('users', {}).items():
                self.users[username] = User(
                    username=username,
                    name=user_data['name'],
                    email=user_data['email'],
                    role=UserRole(user_data['role']),
                    hashed_password=user_data['hashed_password'],
                    is_active=user_data.get('is_active', True)
                )
            logger.info(f"Loaded {len(self.users)} users from {self.users_file}")
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            self.users = {}
    
    def _save_users(self) -> None:
        """Save users to YAML file."""
        try:
            data = {
                'users': {
                    username: {
                        'name': user.name,
                        'email': user.email,
                        'role': user.role.value,
                        'hashed_password': user.hashed_password,
                        'is_active': user.is_active
                    }
                    for username, user in self.users.items()
                }
            }
            
            # Ensure directory exists
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.users_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            logger.info(f"Saved {len(self.users)} users to {self.users_file}")
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        default_password = "admin123"
        hashed_password = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        admin_user = User(
            username="admin",
            name="System Administrator",
            email="admin@ragagent.local",
            role=UserRole.ADMIN,
            hashed_password=hashed_password
        )
        
        self.users["admin"] = admin_user
        self._save_users()
        logger.info("Created default admin user (username: admin, password: admin123)")
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password."""
        if username not in self.users:
            return None
            
        user = self.users[username]
        if not user.is_active:
            return None
            
        if bcrypt.checkpw(password.encode('utf-8'), user.hashed_password.encode('utf-8')):
            return user
        return None
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in ROLE_PERMISSIONS.get(user.role, [])
    
    def create_user(self, username: str, name: str, email: str, password: str, role: UserRole) -> bool:
        """Create a new user."""
        if username in self.users:
            return False
            
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        user = User(
            username=username,
            name=name,
            email=email,
            role=role,
            hashed_password=hashed_password
        )
        
        self.users[username] = user
        self._save_users()
        return True
    
    def update_user(self, username: str, **kwargs) -> bool:
        """Update user information."""
        if username not in self.users:
            return False
            
        user = self.users[username]
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        self._save_users()
        return True
    
    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        if username not in self.users:
            return False
            
        del self.users[username]
        self._save_users()
        return True
    
    def list_users(self) -> List[User]:
        """Get list of all users."""
        return list(self.users.values())

# Global auth manager instance
auth_manager = AuthManager()

def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.error("Please log in to access this feature")
                return None
                
            user = get_current_user()
            if not auth_manager.has_permission(user, permission):
                st.error("You don't have permission to access this feature")
                return None
                
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(required_role: UserRole):
    """Decorator to require specific role."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.error("Please log in to access this feature")
                return None
                
            user = get_current_user()
            if user.role != required_role:
                st.error(f"This feature requires {required_role.value} role")
                return None
                
            return func(*args, **kwargs)
        return wrapper
    return decorator

def login_user(username: str, password: str) -> bool:
    """Login user and store in session."""
    user = auth_manager.authenticate(username, password)
    if user:
        st.session_state.authenticated = True
        st.session_state.user = user
        logger.info(f"User {username} logged in successfully")
        return True
    return False

def logout_user() -> None:
    """Logout current user."""
    if 'user' in st.session_state:
        logger.info(f"User {st.session_state.user.username} logged out")
    st.session_state.authenticated = False
    st.session_state.user = None

def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get('authenticated', False)

def get_current_user() -> Optional[User]:
    """Get current authenticated user."""
    if is_authenticated():
        return st.session_state.get('user')
    return None

def has_permission(permission: Permission) -> bool:
    """Check if current user has permission."""
    user = get_current_user()
    if not user:
        return False
    return auth_manager.has_permission(user, permission)

def render_login_form() -> None:
    """Render login form."""
    st.title("🔐 RAG Agent Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username and password:
                if login_user(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Please enter both username and password")
    
    # Show default credentials for demo
    st.info("**Default Admin Credentials:**\n\nUsername: `admin`\n\nPassword: `admin123`")

def render_user_info() -> None:
    """Render current user information in sidebar."""
    user = get_current_user()
    if user:
        st.sidebar.markdown("---")
        st.sidebar.subheader("👤 User Info")
        st.sidebar.write(f"**Name:** {user.name}")
        st.sidebar.write(f"**Role:** {user.role.value.title()}")
        st.sidebar.write(f"**Email:** {user.email}")
        
        if st.sidebar.button("🚪 Logout"):
            logout_user()
            st.rerun()

def initialize_auth_session():
    """Initialize authentication session state."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None