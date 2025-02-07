import requests

# Configurations
downloadServerUrl = "https://public.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet"
databasketId = "manifest-1599750808610.tcia"
manifestVersion = "3.0"
includeAnnotation = True
noOfrRetry = 4
ListOfSeriesToDownload = [
    "1.2.826.0.1.3680043.2.1125.1.64431440660529413465820250742459468",
    "1.2.826.0.1.3680043.2.1125.1.64196995986655345161142945283707267",
    # Add other series here
]

def download_series(series_id):
    try:
        # Prepare the URL for downloading a particular series
        download_url = f"{downloadServerUrl}?databasketId={databasketId}&manifestVersion={manifestVersion}&seriesUID={series_id}"
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Check if request was successful
        # Assuming data will be stored locally with series UID as filename
        with open(f"{series_id}.tar", 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded {series_id}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {series_id}: {e}")

def main():
    for series_id in ListOfSeriesToDownload:
        download_series(series_id)

if __name__ == "__main__":
    main()
