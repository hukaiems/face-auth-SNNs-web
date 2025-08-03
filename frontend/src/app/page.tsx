"use client";

import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const sendScreenshot = async (base64Image: string[]) => {
  try{
    const response = await axios.post("http://localhost:8000/compare", {
      images: base64Image,
    })
    console.log("Response:", response.data);
  } catch (err: any) {
  console.log("Error details:", err.response?.data);
}
}

export default function Home() {
  const [showCam, setShowCam] = useState(false);
  const webcamRef = useRef<Webcam>(null); // to interact with the webcam instance
  const [screenshot, setScreenshot] = useState<string[]>([]);
  const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));

  useEffect(() => {
    const captureImages = async () => {
      if (showCam) {
        if (webcamRef.current) {
          const captured: string[] = [];
          await delay(2000);
          for (let i = 0; i < 5; i++) {
            const img = webcamRef.current.getScreenshot();
            if (img) {
              captured.push(img);
              console.log(`Captured photo ${i + 1}`);
            }
            await delay(500);
          }
          setScreenshot(captured);
          await sendScreenshot(captured);
          setShowCam(false);
        }
      }
    };
    captureImages();
  }, [showCam]);

  useEffect(() => {
    if (screenshot) {
      setShowCam(false);
    }
  }, [screenshot]);

  return (
    // needs height for centering
    <div className="flex flex-col justify-center items-center h-screen gap-4">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-sky-400 to-indigo-500 text-transparent bg-clip-text p-10 rounded-xl">
        Facial authentication website
      </h1>
      {!showCam && screenshot.length === 0 && (
        <button
          className="border rounded-2xl p-3 hover:bg-sky-700"
          onClick={() => setShowCam(true)}
        >
          Login
        </button>
      )}

      {/* show cam sections */}
      {showCam && (
        <div className="flex flex-col justify-center items-center gap-3">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={{ facingMode: "user" }}
            className="rounded-xl shadow-lg"
          />

          <button
            className="bg-white w-24 text-black rounded-xl h-12 hover:bg-yellow-500"
            onClick={() => {
              setShowCam(false);
              setScreenshot([]);
            }}
          >
            Cancle{" "}
          </button>
        </div>
      )}

      {/* picture screenshot */}
      {screenshot.length > 0 && (
        <div className="flex gap-3">
          {screenshot.map((src, idx) => (
            <img 
              key={idx}
              src={src}
              alt={`Screenshot ${idx + 1}`}
              className="rounded-x max-w-xs"
            />
          ))}
        </div>
      )}
    </div>
  );
}
