import React, { useState, useEffect, useRef } from "react";
import { makeStyles } from "@material-ui/core/styles";

import IconButton from "@material-ui/core/IconButton";
import AttachFileIcon from "@material-ui/icons/AttachFile";
import Grid from "@material-ui/core/Grid";
import {
  userMessage,
  sendMessage,
  sendFile,
  userFile,
} from "../actions/chatbot";
import { connect } from "react-redux";

const useStyles = makeStyles((theme) => ({
  root: {
    "& > *": {
      margin: theme.spacing(1),
    },
  },
  input: {
    display: "none",
  },
}));

const Chat = ({ chat, userMessage, sendMessage, sendFile, ...rest }) => {
  const [message, setMessage] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const endOfMessages = useRef(null);
  const classes = useStyles();
  const scrollToBottom = () => {
    endOfMessages.current.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [chat]);

  //  Function that handles user submission
  const handleClick = async (e) => {
    const code = e.keyCode || e.which;
    if (code == 13) {
      e.preventDefault();
      userMessage(message);
      sendMessage(rest, message);
      setMessage("");
    }
  };

  const handleUploadClick = (event) => {
    var file = event.target.files[0];
    sendFile(rest, file);
  };

  return (
    <div className='container'>
      <h1>AI Chatbot</h1>
      <div className='historyContainer'>
        {chat.length === 0
          ? ""
          : chat.map((msg) => <div className={msg.type}>{msg.message}</div>)}
        <div ref={endOfMessages}></div>
      </div>
      <Grid
        container
        spacing={0}
        direction='row'
        justifyContent='flex-start'
        alignItems='flex-end'
      >
        <Grid item xs={11}>
          <input
            id='chatBox'
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleClick}
            value={message}
          ></input>
        </Grid>
        <Grid item xs={1}>
          <input
            accept='image/*'
            className={classes.input}
            id='icon-button-file'
            type='file'
            onChange={handleUploadClick}
          />
          <label htmlFor='icon-button-file'>
            <IconButton
              color='primary'
              aria-label='upload picture'
              component='span'
            >
              <AttachFileIcon fontSize='small' />
            </IconButton>
          </label>
        </Grid>
      </Grid>
    </div>
  );
};

const mapStateToProps = (state) => ({
  chat: state.chatbot.messages,
});

export default connect(mapStateToProps, {
  userMessage,
  sendMessage,
  sendFile,
})(Chat);
