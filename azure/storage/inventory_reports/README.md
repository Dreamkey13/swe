# Azure Storage Inventory Script

## Overview

This Python script generates detailed inventory spreadsheets for Azure Storage resources, providing comprehensive insights into file shares and blob containers. The inventory captures key information including:

- File and folder names
- Size information (automatically formatted as KB, MB, or GB)
- Item counts
- Creation dates
- Last modified dates

The script supports both Azure File Shares and Azure Blob Containers, with results exported to a single Excel worksheet for easy analysis.

## Purpose

This tool helps administrators and developers to:
- Document storage resource contents
- Track storage utilization
- Audit file and folder timestamps
- Generate reports for billing or compliance purposes
- Easily discover what's consuming space in your storage accounts

## Requirements

- Python 3.6+
- Azure Storage Account with access credentials
- Required Python packages:
  - azure-storage-file-share
  - azure-storage-blob
  - pandas
  - openpyxl

## Setting Up the Environment

### 1. Clone the Repository

```bash
git clone <repository-url>
cd azure-storage-inventory
```

### 2. Virtual Environment (Recommended)

Setting up a virtual environment keeps dependencies isolated:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install azure-storage-file-share azure-storage-blob pandas openpyxl
```

## Azure Connection String

### Obtaining an Azure Connection String

1. **Sign in to the Azure Portal** (https://portal.azure.com)

2. **Navigate to your Storage Account**
   - Select "Storage accounts" from the left menu
   - Choose the storage account you want to inventory

3. **Access Keys**
   - In the storage account menu, scroll down to the "Security + networking" section
   - Select "Access keys"

4. **Copy a Connection String**
   - You'll see two keys, each with its own connection string
   - Click the "Show" button and then "Copy to clipboard" for one of the connection strings

### Setting the Environment Variable

#### On Windows

Command Prompt (session only):
```cmd
set AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
```

PowerShell (session only):
```powershell
$env:AZURE_STORAGE_CONNECTION_STRING = "your_connection_string_here"
```

For permanent storage (user-level):
```powershell
[Environment]::SetEnvironmentVariable("AZURE_STORAGE_CONNECTION_STRING", "your_connection_string_here", "User")
```

#### On macOS/Linux

Session only:
```bash
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string_here"
```

For permanent storage, add to ~/.bashrc, ~/.zshrc, or appropriate shell configuration file:
```bash
echo 'export AZURE_STORAGE_CONNECTION_STRING="your_connection_string_here"' >> ~/.bashrc
source ~/.bashrc
```

## Usage

### Basic Usage

```bash
python azure_storage_inventory.py --storage-type fileshare --name "your_share_name" --output "inventory.xlsx"
```

or

```bash
python azure_storage_inventory.py --storage-type blob --name "your_container_name" --output "inventory.xlsx"
```

### Command Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--storage-type` | Type of storage to inventory (`fileshare` or `blob`) | Yes | - |
| `--name` | Name of the file share or blob container | Yes | - |
| `--output` | Path for the output Excel file | No | azure_storage_inventory.xlsx |
| `--connection-env` | Name of environment variable containing connection string | No | AZURE_STORAGE_CONNECTION_STRING |

### Using a Custom Environment Variable

```bash
# Set custom environment variable
export MY_AZURE_CONNECTION="your_connection_string_here"

# Run with custom environment variable
python azure_storage_inventory.py --storage-type blob --name "my-container" --connection-env MY_AZURE_CONNECTION
```

## Output Format

The script generates an Excel file with a single worksheet named "Inventory" containing:

| Column | Description |
|--------|-------------|
| Name | Full path and name of the file or folder |
| Type | Either "File" or "Folder" |
| Size | Formatted size (B, KB, MB, or GB) |
| ItemCount | Number of items in folder (or 1 for files) |
| CreatedOn | Date and time the item was created |
| LastModified | Date and time the item was last modified |

## Security Considerations

- This script reads Azure connection strings from environment variables to avoid hardcoding credentials
- Never commit connection strings or credentials to source control
- Use Azure Key Vault for production environments
- Consider using service principals with limited permissions for production use

## Troubleshooting

### Common Issues

1. **Missing Environment Variable**
   ```
   Error: Environment variable 'AZURE_STORAGE_CONNECTION_STRING' not found or empty.
   ```
   Solution: Ensure you've set the environment variable as described above.

2. **Unable to Connect to Azure**
   ```
   azure.core.exceptions.ClientAuthenticationError: Authentication failed
   ```
   Solution: Verify your connection string is correct and the storage account exists.

3. **Resource Not Found**
   ```
   Error: Share/Container 'your_name' does not exist.
   ```
   Solution: Check the spelling of your file share or container name.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## License

[Specify license information here]
