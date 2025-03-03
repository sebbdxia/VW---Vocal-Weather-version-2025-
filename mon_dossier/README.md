
# VW - Vocal Weather (version 2025)

## Description
VW - Vocal Weather is an application that provides weather forecasts using voice commands. It leverages FastAPI for the backend, Streamlit for the frontend, and integrates with various services like Azure Speech, Open-Meteo API, PostgreSQL, and Prometheus for metrics collection.

## Features
- Voice command processing for weather forecasts
- Manual input for weather forecasts
- Real-time metrics collection and monitoring
- User feedback collection
- Visualization of weather data and metrics

## Installation
1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```sh
    cd VW - Vocal Weather (version 2025)
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration
1. Create a `.env` file in the root directory and add the following environment variables:
    ```env
    DB_USER=<your-db-user>
    DB_PASSWORD=<your-db-password>
    DB_HOST=<your-db-host>
    DB_NAME=<your-db-name>
    SPEECH_KEY=<your-azure-speech-key>
    SPEECH_REGION=<your-azure-speech-region>
    ```

## Usage
1. Start the FastAPI backend:
    ```sh
    uvicorn main:app --reload
    ```
2. Start the Streamlit frontend:
    ```sh
    streamlit run app.py
    ```

## Endpoints
- `/process_command`: Process voice or manual commands for weather forecasts
- `/metrics`: Expose Prometheus metrics
- `/analysis`: Retrieve logs and feedbacks
- `/top_cities`: Get the number of requests per city
- `/feedback`: Submit user feedback
- `/diagram`: Get the functional architecture diagram

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.
