# RAG Agent RBAC Implementation

## Overview

This document describes the Role-Based Access Control (RBAC) implementation for the RAG Document Assistant application. The RBAC system provides secure user authentication and fine-grained authorization controls.

## Architecture

### Components

1. **Authentication Module** (`src/auth.py`)
   - User management
   - Password hashing with bcrypt
   - Session management
   - Permission checking

2. **User Storage** (`users.yaml`)
   - File-based user database
   - YAML format for easy configuration
   - Encrypted password storage

3. **Enhanced UI** (`src/streamlit_app.py`)
   - Login/logout functionality
   - Role-based tab visibility
   - Permission-gated features

### User Roles

#### 1. Admin Role
- **Full system access**
- **Permissions:**
  - View documents
  - Search documents  
  - Upload documents
  - Delete documents
  - View system status
  - Manage users
  - Configure system
- **UI Access:**
  - Document Q&A tab
  - Document Ingestion tab
  - System Status tab
  - User Management tab
  - Full sidebar information

#### 2. User Role  
- **Standard user access**
- **Permissions:**
  - View documents
  - Search documents
  - Upload documents
  - View system status
- **UI Access:**
  - Document Q&A tab
  - Document Ingestion tab
  - System Status tab
  - System information in sidebar

#### 3. Viewer Role
- **Read-only access**
- **Permissions:**
  - View documents
  - Search documents
- **UI Access:**
  - Document Q&A tab only
  - Basic user info in sidebar

## Security Features

### Authentication
- **Password Hashing:** bcrypt with salt
- **Session Management:** Streamlit session state
- **Default Admin:** Auto-created on first run

### Authorization
- **Permission-Based:** Granular permission system
- **Role-Based:** Predefined role templates
- **UI Enforcement:** Dynamic tab and feature visibility

### User Management
- **Self-Service:** Login/logout
- **Admin Controls:** User creation, deletion, role assignment
- **Secure Storage:** Encrypted passwords, file-based persistence

## Default Credentials

**Admin User:**
- Username: `admin`
- Password: `admin123`
- Role: Admin

**Note:** Change the default admin password immediately after deployment.

## Usage

### For End Users

1. **Login:**
   - Navigate to the application
   - Enter username and password
   - Access features based on your role

2. **Role-Specific Features:**
   - **Admin:** Full access to all features
   - **User:** Standard document operations
   - **Viewer:** Read-only document access

### For Administrators

1. **User Management:**
   - Login as admin
   - Navigate to "User Management" tab
   - Add/remove users
   - Assign roles

2. **System Configuration:**
   - Access system status
   - Monitor usage
   - Configure application settings

## Implementation Details

### Permission System

```python
class Permission(Enum):
    VIEW_DOCUMENTS = "view_documents"
    SEARCH_DOCUMENTS = "search_documents"
    UPLOAD_DOCUMENTS = "upload_documents"
    DELETE_DOCUMENTS = "delete_documents"
    VIEW_SYSTEM_STATUS = "view_system_status"
    MANAGE_USERS = "manage_users"
    CONFIGURE_SYSTEM = "configure_system"
```

### Role Permissions Mapping

```python
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [all permissions],
    UserRole.USER: [limited permissions],
    UserRole.VIEWER: [minimal permissions],
}
```

### UI Permission Checks

```python
@require_permission(Permission.UPLOAD_DOCUMENTS)
def render_ingestion_tab():
    # Only accessible to users with upload permission
    pass
```

## File Structure

```
rag-agent/
├── src/
│   ├── auth.py              # RBAC implementation
│   ├── streamlit_app.py     # Enhanced UI with RBAC
│   └── ...
├── users.yaml              # User database
├── test_rbac.py            # RBAC tests
└── RBAC_GUIDE.md           # This documentation
```

## Security Considerations

1. **Password Security:**
   - Use strong passwords
   - Change default credentials
   - Consider password complexity requirements

2. **Session Security:**
   - Sessions timeout on browser close
   - Logout clears session data

3. **File Security:**
   - Secure `users.yaml` file permissions
   - Consider database backend for production

4. **Network Security:**
   - Use HTTPS in production
   - Secure session cookies

## Testing

Run the RBAC test suite:

```bash
python test_rbac.py
```

Tests cover:
- User authentication
- Permission checking
- Role-based access
- User management operations

## Future Enhancements

1. **Database Backend:** Replace file-based storage
2. **Password Policies:** Enforce complexity requirements
3. **Session Timeout:** Automatic logout after inactivity
4. **Audit Logging:** Track user actions
5. **OAuth Integration:** Support external authentication providers
6. **Multi-Factor Authentication:** Enhanced security
7. **API Access Control:** Extend RBAC to API endpoints