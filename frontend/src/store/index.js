import { combineReducers } from "redux";

import chatbotReducer from "./chatbotReducer";
import Reducer from "./authReducer";

export default combineReducers({ chatbotReducer, Reducer });