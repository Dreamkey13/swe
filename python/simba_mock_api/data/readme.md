# Data Directory

## Purpose
This directory contains the JSON files that are served by the Simba Mock API. These files provide predefined responses for different request IDs, allowing for consistent and predictable API behavior during agent development and testing.

## File Organization

### Naming Convention
Files in this directory should follow this naming convention:
- `request_[ID].json` - where `[ID]` is a numeric identifier

For example:
- `request_1.json`
- `request_2.json`
- `request_999.json`

The numeric ID in the filename corresponds to the `requestId` parameter that is passed to the `/simbarequest` endpoint.

### File Format
Each JSON file should contain a valid JSON object. The structure of the JSON is flexible, but a recommended format is:

```json
{
  "message": "Description of this response",
  "status": "success",
  "timestamp": "ISO-8601 timestamp",
  "data": {
    // Your specific response data here
  }
}
```

## How to Add New Mock Responses

1. Create a new JSON file following the naming convention (`request_[ID].json`)
2. Structure your JSON data according to your requirements
3. Place the file in this directory
4. The file will be automatically accessible via the `/simbarequest` endpoint with the corresponding request ID

For example, if you create a file named `request_42.json`, it will be accessible via:
```
GET /simbarequest?request_id=42
```

## Example Files
The directory contains several example files:

### request_1.json
Basic response with a simple data structure.

### request_2.json
More complex response with nested arrays and objects.

## Technical Details

- Files are read using standard Python file I/O operations
- JSON parsing is handled using Python's built-in `json` module
- If a file doesn't exist for a given request ID, the API will return a 404 error
- The API doesn't cache responses, so any changes to the JSON files will be immediately reflected in the API responses

## Best Practices

1. **Consistent Structure**: Try to maintain a consistent structure across different response files
2. **Valid JSON**: Ensure your JSON files contain valid JSON (you can use tools like JSONLint to validate)
3. **Meaningful Data**: Include realistic data that represents actual use cases
4. **Documentation**: If your mock responses become complex, consider including comments in this README about specific files
5. **Organization**: For large numbers of files, consider creating subdirectories and updating the API code to handle them
