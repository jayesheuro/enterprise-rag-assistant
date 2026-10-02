Title: API Design Standards
Version: 2.0
Last Updated: 2024-01-22
Owner: Core Platform Team

# API Design Standards

These guidelines apply to all internal and external RESTful APIs developed at Meridian Technologies Inc.

## 1. REST Conventions
* Use plural nouns for resources (e.g., `/users`, not `/user`).
* Use HTTP methods correctly: `GET` for retrieval, `POST` for creation, `PUT` for full updates, `PATCH` for partial updates, and `DELETE` for removal.
* Never use verbs in the URL path. Actions should be represented by the HTTP method applied to a resource.

## 2. Versioning
* All APIs must be versioned.
* Versioning must be implemented in the URL path (e.g., `/api/v1/users`).
* Deprecation of an API version requires a minimum of 6 months' notice to consumers before the endpoint is turned off.

## 3. Authentication
* **Internal APIs:** Must be secured using OAuth 2.0 with JWT (JSON Web Tokens) issued by our internal Okta identity provider.
* **External/Public APIs:** Must require a Meridian API Key passed in the `X-API-Key` HTTP header.

## 4. Rate Limiting
To protect backend services, all public-facing APIs must implement rate limiting.
* The standard rate limit is 100 requests per minute per API key.
* Endpoints returning paginated data are limited to 50 requests per minute.
* Rate limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) must be included in all responses.

## 5. Error Response Format
When an API encounters an error, it must return an appropriate HTTP status code (e.g., 400 for bad request, 401 for unauthorized, 404 for not found, 500 for internal server error) and a standardized JSON error payload:

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "The 'email' field is required.",
    "details": [
      {
        "field": "email",
        "reason": "missing"
      }
    ]
  }
}
```

## 6. OpenAPI Specification
Every API must be documented using the OpenAPI 3.0 specification.
* The `openapi.yaml` file must be stored in the root directory of the service's repository.
* The CI/CD pipeline will automatically validate the OpenAPI spec and publish it to the central Meridian Developer Portal.
