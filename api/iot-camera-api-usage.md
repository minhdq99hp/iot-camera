## POST Request
`curl -X POST --form "fileInput=@[FILE_PATH]" [BASE_URL]/upload_data --output [OUTPUT_FILE_PATH]`

Example: `curl -X POST --form "fileInput=@/home/minhdq99hp/Desktop/hello.jpg" http://127.0.0.1:5000/upload_data --output /home/minhdq99hp/Desktop/output.jpg`
