# API Usage Examples

## Python

### Using requests

```python
import requests

# Transcribe audio file
url = "http://localhost:8000/v1/audio/transcriptions"

with open("audio.wav", "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    data = {"model": "parakeet-tdt-0.6b-v3"}
    response = requests.post(url, files=files, data=data)

transcript = response.json()
print(transcript["text"])
```

### Using OpenAI Python client

```python
from openai import OpenAI

# Configure client to use local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Transcribe audio
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="parakeet-tdt-0.6b-v3",
        file=audio_file,
        response_format="text"
    )

print(transcript)
```

### Async example with aiohttp

```python
import aiohttp
import asyncio

async def transcribe_audio(audio_path: str):
    url = "http://localhost:8000/v1/audio/transcriptions"

    async with aiohttp.ClientSession() as session:
        with open(audio_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename="audio.wav")
            data.add_field("model", "parakeet-tdt-0.6b-v3")

            async with session.post(url, data=data) as response:
                result = await response.json()
                return result["text"]

# Run
text = asyncio.run(transcribe_audio("audio.wav"))
print(text)
```

## JavaScript / Node.js

### Using fetch

```javascript
const fs = require('fs');
const FormData = require('form-data');

async function transcribeAudio(audioPath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(audioPath));
    form.append('model', 'parakeet-tdt-0.6b-v3');

    const response = await fetch('http://localhost:8000/v1/audio/transcriptions', {
        method: 'POST',
        body: form
    });

    const result = await response.json();
    return result.text;
}

transcribeAudio('audio.wav').then(console.log);
```

### Using axios

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function transcribeAudio(audioPath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(audioPath));
    form.append('model', 'parakeet-tdt-0.6b-v3');

    const response = await axios.post(
        'http://localhost:8000/v1/audio/transcriptions',
        form,
        { headers: form.getHeaders() }
    );

    return response.data.text;
}

transcribeAudio('audio.wav')
    .then(text => console.log(text))
    .catch(err => console.error(err));
```

## cURL

### Basic transcription

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=parakeet-tdt-0.6b-v3"
```

### Get plain text response

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=parakeet-tdt-0.6b-v3" \
  -F "response_format=text"
```

### Get SRT subtitles

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=parakeet-tdt-0.6b-v3" \
  -F "response_format=srt" \
  -o subtitles.srt
```

## Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
)

type TranscriptionResponse struct {
    Text string `json:"text"`
}

func transcribeAudio(audioPath string) (string, error) {
    // Open audio file
    file, err := os.Open(audioPath)
    if err != nil {
        return "", err
    }
    defer file.Close()

    // Create multipart form
    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)

    // Add file
    part, err := writer.CreateFormFile("file", audioPath)
    if err != nil {
        return "", err
    }
    io.Copy(part, file)

    // Add model field
    writer.WriteField("model", "parakeet-tdt-0.6b-v3")
    writer.Close()

    // Send request
    req, err := http.NewRequest("POST", "http://localhost:8000/v1/audio/transcriptions", body)
    if err != nil {
        return "", err
    }
    req.Header.Set("Content-Type", writer.FormDataContentType())

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    // Parse response
    var result TranscriptionResponse
    json.NewDecoder(resp.Body).Decode(&result)

    return result.Text, nil
}

func main() {
    text, err := transcribeAudio("audio.wav")
    if err != nil {
        panic(err)
    }
    fmt.Println(text)
}
```

## Rust

```rust
use reqwest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct TranscriptionResponse {
    text: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // Read audio file
    let audio_bytes = std::fs::read("audio.wav")?;

    // Create form
    let form = reqwest::multipart::Form::new()
        .part("file", reqwest::multipart::Part::bytes(audio_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")?)
        .text("model", "parakeet-tdt-0.6b-v3");

    // Send request
    let response = client
        .post("http://localhost:8000/v1/audio/transcriptions")
        .multipart(form)
        .send()
        .await?;

    // Parse response
    let result: TranscriptionResponse = response.json().await?;
    println!("{}", result.text);

    Ok(())
}
```

## C#

```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json;

class Program
{
    class TranscriptionResponse
    {
        public string Text { get; set; }
    }

    static async Task<string> TranscribeAudio(string audioPath)
    {
        using (var client = new HttpClient())
        using (var form = new MultipartFormDataContent())
        {
            // Add file
            var fileContent = new ByteArrayContent(File.ReadAllBytes(audioPath));
            fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("audio/wav");
            form.Add(fileContent, "file", "audio.wav");

            // Add model
            form.Add(new StringContent("parakeet-tdt-0.6b-v3"), "model");

            // Send request
            var response = await client.PostAsync("http://localhost:8000/v1/audio/transcriptions", form);
            var json = await response.Content.ReadAsStringAsync();

            // Parse response
            var result = JsonConvert.DeserializeObject<TranscriptionResponse>(json);
            return result.Text;
        }
    }

    static async Task Main(string[] args)
    {
        string text = await TranscribeAudio("audio.wav");
        Console.WriteLine(text);
    }
}
```

## PHP

```php
<?php

function transcribeAudio($audioPath) {
    $url = 'http://localhost:8000/v1/audio/transcriptions';

    $ch = curl_init();

    $postData = [
        'file' => new CURLFile($audioPath, 'audio/wav', 'audio.wav'),
        'model' => 'parakeet-tdt-0.6b-v3'
    ];

    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

    $response = curl_exec($ch);
    curl_close($ch);

    $result = json_decode($response, true);
    return $result['text'];
}

$text = transcribeAudio('audio.wav');
echo $text . "\n";

?>
```

## Ruby

```ruby
require 'httparty'

def transcribe_audio(audio_path)
  url = 'http://localhost:8000/v1/audio/transcriptions'

  response = HTTParty.post(url, {
    body: {
      file: File.open(audio_path),
      model: 'parakeet-tdt-0.6b-v3'
    }
  })

  JSON.parse(response.body)['text']
end

text = transcribe_audio('audio.wav')
puts text
```

## Java

```java
import java.io.File;
import java.io.IOException;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.json.JSONObject;

public class ParakeetClient {
    public static String transcribeAudio(String audioPath) throws IOException {
        CloseableHttpClient client = HttpClients.createDefault();
        HttpPost post = new HttpPost("http://localhost:8000/v1/audio/transcriptions");

        // Build multipart form
        HttpEntity entity = MultipartEntityBuilder.create()
            .addBinaryBody("file", new File(audioPath))
            .addTextBody("model", "parakeet-tdt-0.6b-v3")
            .build();

        post.setEntity(entity);

        // Send request
        CloseableHttpResponse response = client.execute(post);
        String json = EntityUtils.toString(response.getEntity());

        // Parse response
        JSONObject obj = new JSONObject(json);
        return obj.getString("text");
    }

    public static void main(String[] args) throws IOException {
        String text = transcribeAudio("audio.wav");
        System.out.println(text);
    }
}
```

## Batch Processing Example (Python)

```python
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

def transcribe_file(audio_path: Path) -> dict:
    """Transcribe a single audio file"""
    url = "http://localhost:8000/v1/audio/transcriptions"

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        data = {"model": "parakeet-tdt-0.6b-v3"}
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()

    return {
        "file": audio_path.name,
        "text": response.json()["text"]
    }

def batch_transcribe(audio_dir: str, max_workers: int = 4):
    """Transcribe all audio files in a directory"""
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav"))

    print(f"Found {len(audio_files)} audio files")
    print(f"Processing with {max_workers} workers...")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(transcribe_file, audio_files))

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Average: {elapsed/len(audio_files):.2f} seconds per file")

    return results

# Example usage
results = batch_transcribe("./audio_files", max_workers=4)

for result in results:
    print(f"\n{result['file']}:")
    print(f"  {result['text']}")
```
