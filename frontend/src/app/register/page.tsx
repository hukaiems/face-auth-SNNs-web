"use client";
import Webcam from "react-webcam";
import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL!; //not null or undefined

interface RegisterResponse {
  status: string;
  user_id: string;
  registered_poses: string[];
}

export default function Register() {
  const [userName, setUserName] = useState<string>("");
  const [showCam, setShowCam] = useState<boolean>(false);
  const webcamRef = useRef<Webcam>(null);
  const [instruction, setInstruction] = useState<string>("");
  const [screenShot, setScreenShot] = useState<string[]>([]);
  const [serverResponse, setServerResponse] = useState<RegisterResponse | null>(
    null
  );
  const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));
  const router = useRouter();

  // handle userName change
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setUserName(e.target.value);
  }

  function handleBlur() {
    if (userName.trim()) {
      //check to see if user type something after trim
      setShowCam(true);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && userName.trim()) {
      (e.target as HTMLInputElement).blur();
      setShowCam(true);
    }
  }

  const sendData = async (
    images: string[],
    userName: string
  ): Promise<RegisterResponse | null> => {
    try {
      const formData = new FormData();
      formData.append("user_id", userName);

      const poseNames = ["frontal", "left", "right"];
      for (let i = 0; i < 3; i++) {
        const base64 = images[i];
        const blob = await (await fetch(base64)).blob();
        formData.append(poseNames[i], blob, `${poseNames[i]}.jpg`);
      }

      const response = await axios.post<RegisterResponse>(
        `${API_BASE}/register`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      console.log("Backend response:", response.data);
      return response.data;
    } catch{
      setShowCam(false);
      return null;
    }
  };

  //   interact with camera after showCam change
  useEffect(() => {
    const registerImages = async () => {
      if (showCam && webcamRef.current) {
        const captured: string[] = [];
        setInstruction("Put your face in the circle ");
        await delay(4000);
        // first shot
        setInstruction("Look in front of the camera");
        await delay(4000);
        let img = webcamRef.current.getScreenshot();
        if (img) {
          captured.push(img);
          console.log(`Captured photo!`);
        }
        // second shot
        setInstruction(" Lean your face a little bit to the left");
        await delay(4000);
        img = webcamRef.current.getScreenshot();
        if (img) {
          captured.push(img);
          console.log(`Captured photo!`);
        }
        // third shot
        setInstruction(" Lean your face a little bit to the right");
        await delay(4000);

        img = webcamRef.current.getScreenshot();
        if (img) {
          captured.push(img);
          console.log(`Captured photo!`);
        }
        setScreenShot(captured);
        const result = await sendData(captured, userName);
        if (result) {
          setServerResponse(result);
        }
      }
    };

    registerImages();
  }, [showCam, userName]);

  const goToHome = () => {
    router.push("/");
  };

  return (
    // main div
    <div>
      {/* input section */}
      {!showCam ? (
        <div className="flex flex-col justify-center items-center h-screen gap-4">
          {screenShot.length > 0 && !serverResponse && (
            <h1>
              Registration failed, move your face closer to the camera next
              time, try again.
            </h1>
          )}
          <h1 className="text-4xl">Enter your name below</h1>
          <input
            placeholder="Enter your name"
            value={userName}
            onChange={handleChange}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            className="border rounded-lg"
            disabled={showCam} //stop people to change input after show the Cam
          ></input>
        </div>
      ) : (
        <div className="flex flex-col justify-center items-center gap-3">
          <h1 className="text-4xl">{instruction}</h1>
          <div className="relative rounded-xl">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={{ facingMode: "user" }}
              className="shadow-lg rounded-xl"
            />
            {/* 📌 Overlay: absolute, no pointer events */}
            <div className="absolute inset-0 flex justify-center items-center pointer-events-none">
              {/* adjust w-3/4 h-3/4 to control size of the oval */}
              <div className="border-4 border-white rounded-full w-2/4 h-4/4" />
            </div>
          </div>
        </div>
      )}

      {serverResponse && (
        <div className="flex flex-col gap-3 pt-4 justify-center items-center">
          <h1>User registered as: {serverResponse.user_id}</h1>
          <button
            className="border p-2 rounded-xl hover:bg-gray-100/40"
            onClick={goToHome}
          >
            Back to home page
          </button>
        </div>
      )}
    </div>
  );
}
