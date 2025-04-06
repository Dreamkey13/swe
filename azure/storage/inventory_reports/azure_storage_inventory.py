import os
from azure.storage.fileshare import (
    ShareServiceClient,
    FileProperties,
)
from azure.storage.blob import BlobServiceClient
import pandas as pd
import argparse


def get_size_formatted(size_bytes):
    """Convert bytes to human-readable format (KB, MB, GB)"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.2f} MB"
    else:
        return f"{size_bytes/(1024**3):.2f} GB"


def process_file_share(connection_string, share_name, output_file):
    """Process an Azure File Share and create inventory spreadsheet"""
    service_client = ShareServiceClient.from_connection_string(connection_string)
    share_client = service_client.get_share_client(share_name)

    # Check if share exists
    if not share_client.share_name:
        print(f"Error: Share '{share_name}' does not exist.")
        return False

    # Get the root directory client
    root_dir = share_client.get_directory_client("")

    # Lists to store results
    inventory_data = []

    # Process root folder first
    process_file_share_directory(root_dir, "", inventory_data)

    # Create dataframe
    inventory_df = pd.DataFrame(
        inventory_data,
        columns=["Name", "Type", "Size", "ItemCount", "CreatedOn", "LastModified"],
    )

    # Save to Excel
    inventory_df.to_excel(output_file, sheet_name="Inventory", index=False)

    print(
        f"Inventory for file share '{share_name}' created successfully: {output_file}"
    )
    return True


def process_file_share_directory(dir_client, path, inventory_data):
    # Only process root level items
    for item in dir_client.list_directories_and_files():
        if isinstance(item, FileProperties):  # If it's a file in root
            # Add single file entry
            file_client = dir_client.get_file_client(item.name)
            file_properties = file_client.get_file_properties()
            inventory_data.append(
                [
                    item.name,
                    "file",
                    get_size_formatted(file_properties.size),
                    1,
                    (
                        file_properties.creation_time.replace(tzinfo=None)
                        if file_properties.creation_time
                        else ""
                    ),
                    (
                        file_properties.last_modified.replace(tzinfo=None)
                        if file_properties.creation_time
                        else ""
                    ),
                ]
            )
        else:  # It's a folder
            # Process folder contents to aggregate data
            subdir_client = dir_client.get_subdirectory_client(item.name)
            folder_size, folder_files_count = aggregate_folder_contents(subdir_client)

            inventory_data.append(
                [
                    item.name,
                    "folder",
                    get_size_formatted(folder_size),
                    folder_files_count,
                    (
                        item.creation_time.replace(tzinfo=None)
                        if item.creation_time
                        else ""
                    ),
                    (
                        item.last_modified.replace(tzinfo=None)
                        if item.creation_time
                        else ""
                    ),
                    # Other folder properties
                ]
            )
    return inventory_data


def aggregate_folder_contents(dir_client, current_path=""):
    """
    Recursively process folder contents and return aggregated metrics
    """
    total_size = 0
    total_files = 0

    for item in dir_client.list_directories_and_files():
        if isinstance(item, FileProperties):  # If it's a file
            file_client = dir_client.get_file_client(item.name)
            file_properties = file_client.get_file_properties()
            total_size += file_properties.size
            total_files += 1
        else:  # It's a subfolder
            subdir_client = dir_client.get_subdirectory_client(item.name)
            new_path = f"{current_path}/{item.name}" if current_path else item.name
            size, files = aggregate_folder_contents(subdir_client, new_path)
            total_size += size
            total_files += files

    return total_size, total_files


def process_blob_container(connection_string, container_name, output_file):
    """Process an Azure Blob Storage container and create inventory spreadsheet"""
    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)

    # Check if container exists
    try:
        container_client.get_container_properties()
    except:
        print(f"Error: Container '{container_name}' does not exist.")
        return False

    # Get all blobs including those in virtual folders
    all_blobs = list(container_client.list_blobs(include=["metadata"]))

    # List to store all inventory data
    inventory_data = []

    # Dictionary to store virtual folder information
    virtual_folders = {}

    # Process all blobs
    for blob in all_blobs:
        blob_name = blob.name
        # Check if blob is in a virtual folder
        if "/" in blob_name:
            folder_path = os.path.dirname(blob_name)
            file_name = os.path.basename(blob_name)

            # Add or update folder info
            folders = []
            current_path = ""
            for folder in folder_path.split("/"):
                current_path = f"{current_path}/{folder}" if current_path else folder
                folders.append(current_path)

            for folder in folders:
                if folder not in virtual_folders:
                    virtual_folders[folder] = {
                        "total_size": 0,
                        "item_count": 0,
                        "created_on": blob.creation_time,
                        "last_modified": blob.last_modified,
                    }

                virtual_folders[folder]["total_size"] += blob.size
                virtual_folders[folder]["item_count"] += 1

                # Update folder timestamps
                if blob.creation_time < virtual_folders[folder]["created_on"]:
                    virtual_folders[folder]["created_on"] = blob.creation_time
                if blob.last_modified > virtual_folders[folder]["last_modified"]:
                    virtual_folders[folder]["last_modified"] = blob.last_modified
        else:
            # This is a blob in the root
            file_name = blob_name

        # Add file data to inventory
        inventory_data.append(
            [
                blob_name,
                "File",
                get_size_formatted(blob.size),
                1,  # Item count for files is always 1
                (
                    blob.creation_time.strftime("%Y-%m-%d %H:%M:%S")
                    if blob.creation_time
                    else "Unknown"
                ),
                (
                    blob.last_modified.strftime("%Y-%m-%d %H:%M:%S")
                    if blob.last_modified
                    else "Unknown"
                ),
            ]
        )

    # Add folder data to inventory
    for folder_name, folder_info in virtual_folders.items():
        inventory_data.append(
            [
                folder_name,
                "Folder",
                get_size_formatted(folder_info["total_size"]),
                folder_info["item_count"],
                (
                    folder_info["created_on"].strftime("%Y-%m-%d %H:%M:%S")
                    if folder_info["created_on"]
                    else "Unknown"
                ),
                (
                    folder_info["last_modified"].strftime("%Y-%m-%d %H:%M:%S")
                    if folder_info["last_modified"]
                    else "Unknown"
                ),
            ]
        )

    # Create dataframe
    inventory_df = pd.DataFrame(
        inventory_data,
        columns=["Name", "Type", "Size", "ItemCount", "CreatedOn", "LastModified"],
    )

    # Save to Excel
    inventory_df.to_excel(output_file, sheet_name="Inventory", index=False)

    print(
        f"Inventory for blob container '{container_name}' created successfully: {output_file}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate inventory spreadsheet for Azure Storage"
    )
    parser.add_argument(
        "--storage-type",
        required=True,
        choices=["fileshare", "blob"],
        help="Type of storage (fileshare or blob)",
    )
    parser.add_argument(
        "--name", required=True, help="Name of file share or blob container"
    )
    parser.add_argument(
        "--output",
        default="azure_storage_inventory.xlsx",
        help="Output Excel file path (default: azure_storage_inventory.xlsx)",
    )
    parser.add_argument(
        "--connection-env",
        default="AZURE_STORAGE_CONNECTION_STRING",
        help="Environment variable name containing the connection string (default: AZURE_STORAGE_CONNECTION_STRING)",
    )

    args = parser.parse_args()

    # Get the connection string from the environment variable
    connection_string = os.environ.get(args.connection_env)
    if not connection_string:
        print(
            f"Error: Environment variable '{args.connection_env}' not found or empty."
        )
        print(
            f"Please set it with: export {args.connection_env}='your_connection_string'"
        )
        return False

    if args.storage_type == "fileshare":
        process_file_share(connection_string, args.name, args.output)
    else:  # blob
        process_blob_container(connection_string, args.name, args.output)


if __name__ == "__main__":
    main()
