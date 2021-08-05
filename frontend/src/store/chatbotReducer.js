//import type
import {
  INPUT_SUCCESS,
  INPUT_FAIL,
  MESSAGE_SUCCESS,
  MESSAGE_FAIL,
} from "../actions/types";

//Initial state
const initialState = {
  messages: [],
};

export default (state = initialState, action) => {
  const { type, payload } = action;
  let { messages } = state;
  switch (type) {
    case INPUT_SUCCESS:
      messages = [...messages, { message: payload, type: "user" }];
      return { ...state, messages };
    case INPUT_FAIL:
      return {
        ...state,
      };
    case MESSAGE_SUCCESS:
      messages = [...messages, { message: payload, type: "bot" }];
      return {
        ...state,
        messages,
      };
    case MESSAGE_FAIL:
      return {
        ...state,
      };
    default:
      return {
        ...state,
      };
  }
};
