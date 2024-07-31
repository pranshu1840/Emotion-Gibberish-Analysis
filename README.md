
Steps for Docker Setup


1. Build the Docker Image by running below command in Vscode terminal while in Emotion_Gibberish directory:

   
   docker build -t emotion-gibberish-app .
   

2. Run the Docker Container by running below command:

   
   docker run -p 5000:5000 emotion-gibberish-app
   

3. Access the Web UI:

   Open your web browser and go to `http://localhost:5000`.
