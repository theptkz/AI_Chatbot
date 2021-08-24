//import types
import {
  INPUT_SUCCESS,
  INPUT_FAIL,
  MESSAGE_FAIL,
  MESSAGE_SUCCESS,
  FILE_SUCCESS,
  FILE_FAIL,
} from "./types";

import axios from "axios";
import * as settings from "../settings";
//function that handle users messagese
export const userMessage = (message) => async (dispatch) => {
  try {
    dispatch({ type: INPUT_SUCCESS, payload: message });
  } catch (err) {
    dispatch({ type: INPUT_FAIL });
  }
};

export const sendMessage = (rest, message) => async (dispatch) => {
  let headers = { Authorization: `Token ${rest.token}` };
  let url = settings.API_SERVER + "/api/predict/";
  let method = "post";
  let config = { headers, method, url, data: { message: message } };
  axios(config)
    .then((res) => {
      dispatch({
        type: MESSAGE_SUCCESS,
        payload: `Intent: ${res.data.Intent}, Entity: ${res.data.Entity}`,
      });
    })
    .catch((error) => {
      dispatch({ type: MESSAGE_FAIL });
    });
};

export const sendFile = (rest, uploadfile) => async (dispatch) => {
  console.log(uploadfile);
  const formData = new FormData();
  formData.append("file", uploadfile);
  let headers = {
    Authorization: `Token ${rest.token}`,
  };
  let url = settings.API_SERVER + "/api/upload/";
  let method = "post";
  let config = { headers, method, url, data: formData };
  axios(config)
    .then((res) => {
      console.log(res);
      dispatch({
        type: FILE_SUCCESS,
        payload: `File: ${res.data.file}`,
      });
    })
    .catch((error) => {
      dispatch({ type: FILE_FAIL });
    });
};
