## POST Request
`curl -X POST -F "file_input=@[FILE_PATH]" -F "output_type=output_file" --output [OUTPUT_FILE_PATH]`

Example: 

`curl -X POST -F "file_input=@[FILE_PATH]" -F "output_type=output_file" https://817a9471.ngrok.io/streaming`


## Streaming
Streaming online your video.


### POST request
This command will return a `streaming_id`.

#### For streaming a video file
`curl -X POST -F "file_input=@[FILE_PATH]" https://817a9471.ngrok.io/streaming`
#### For streaming using RTSP
`curl -X POST -F "rtsp_url=[YOUR_RTSP_URL]" https://817a9471.ngrok.io/streaming`


### GET request
`curl https://817a9471.ngrok.io/streaming?streaming_id=[YOUR_STREAMING_ID]`

