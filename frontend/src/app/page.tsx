"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from 'next/navigation';
import Webcam from "react-webcam";
import axios from "axios";

export interface RecognizedResponse {
  recognized: boolean;
  user_id: string | null;
  similarity: number;
  votes: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL!;
if (!API_BASE) throw new Error("NEXT_PUBLIC_API_BASE_URL is not defined");

export default function Home() {
  const [showCam, setShowCam] = useState(false);
  const webcamRef = useRef<Webcam>(null); // to interact with the webcam instance
  const [screenshot, setScreenshot] = useState<string[]>([]);
  const [result, setResult] = useState<RecognizedResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));
  const router = useRouter();

  const sendScreenshot = async (
    base64Image: string[]
  ): Promise<RecognizedResponse | null> => {
    try {
      const response = await axios.post<RecognizedResponse>(
        `${API_BASE}/compare`,
        {
          images: base64Image,
        },
        { headers: { "Content-Type": "application/json" } }
      );
      console.log("Response:", response.data);

      return response.data;
    } catch (err: any) {
      console.log("Error details:", err.response?.data);
      const detail = err.response?.data?.detail ?? "Unknown error";
      setErrorMsg(detail);
      return null;
    }
  };

  useEffect(() => {
    const captureImages = async () => {
      if (showCam) {
        if (webcamRef.current) {
          const captured: string[] = [];
          await delay(5000);
          for (let i = 0; i < 5; i++) {
            const img = webcamRef.current.getScreenshot();
            if (img) {
              captured.push(img);
              console.log(`Captured photo ${i + 1}`);
            }
            await delay(500);
          }
          setScreenshot(captured);
          const res = await sendScreenshot(captured);
          setShowCam(false);
          setResult(res);
        }
      }
    };
    captureImages();
  }, [showCam]);

  const goToRegister = () => {
    router.push('/register');
  };

  return (
    // needs height for centering
    <div className="flex flex-col justify-center items-center h-screen gap-4">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-sky-400 to-indigo-500 text-transparent bg-clip-text p-10 pb-0 rounded-xl">
        Facial authentication website
      </h1>
      {!showCam && screenshot.length === 0 && (
        <div className="flex gap-3">
          <button
            className="border rounded-2xl p-3 hover:bg-sky-700"
            onClick={() => setShowCam(true)}
          >
            Login
          </button>
          <button
            className="border rounded-2xl p-3 hover:bg-sky-700"
            onClick={goToRegister}
          >
            Register
          </button>
        </div>
      )}

      {/* show cam sections */}
      {showCam && (
        <div className="flex flex-col justify-center items-center gap-3">
          <h1 className="text-4xl">Put your face in the circle</h1>
          <div className="relative rounded-xl">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={{ facingMode: "user" }}
              className="shadow-lg"
            />
            {/* 📌 Overlay: absolute, no pointer events */}
            <div className="absolute inset-0 flex justify-center items-center pointer-events-none">
              {/* adjust w-3/4 h-3/4 to control size of the oval */}
              <div className="border-4 border-white rounded-full w-2/4 h-4/4" />
            </div>
          </div>

          <button
            className="bg-white w-24 text-black rounded-xl h-12 hover:bg-yellow-500"
            onClick={() => {
              setShowCam(false);
              setScreenshot([]);
            }}
          >
            Cancel
          </button>
        </div>
      )}

      {/* Error face too far */}
      {errorMsg && <h1>{errorMsg}</h1>}

      {/* picture screenshot  and authentication result*/}
      {screenshot.length > 0 && (
        <div className="flex gap-3">
          <img
            src={screenshot[3]}
            alt="Screenshot 1"
            className="rounded-xl max-w-xs"
          />
          {result?.recognized === true && (
            <div>
              <h1>Name: {result.user_id}</h1>
              <h1>Similarity score: {result?.similarity}</h1>
              <h1>Votes: {result.votes}</h1>
            </div>
          )}

          {result?.recognized === false && (
            <div>
              <h1>Can't recognized user.</h1>
              <h1>Similarity score: {result?.similarity}</h1>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
