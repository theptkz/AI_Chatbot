import React from "react";
import { BrowserRouter, Redirect, Route, Switch } from "react-router-dom";

import Login from "./components/Login";
import Chat from "./components/Chat";
import PasswordUpdate from "./components/PasswordUpdate";

// A wrapper for <Route> that redirects to the login screen if you're not yet authenticated.
function PrivateRoute({ isAuthenticated, children, ...rest }) {
  return (
    <Route
      {...rest}
      render={({ location }) =>
        isAuthenticated ? (
          children
        ) : (
          <Redirect
            to={{
              pathname: "/login/",
              state: { from: location },
            }}
          />
        )
      }
    />
  );
}

function Urls(props) {
  return (
    <div>
      <BrowserRouter>
        <Switch>
          <Route exact path='/login/'>
            <Login {...props} />
          </Route>
          <PrivateRoute exact path='/' isAuthenticated={props.isAuthenticated}>
            <Chat {...props} className='container' />
          </PrivateRoute>
          <PrivateRoute
            exact
            path='/update_password/'
            isAuthenticated={props.isAuthenticated}
          >
            <PasswordUpdate {...props} />
          </PrivateRoute>
        </Switch>
      </BrowserRouter>
    </div>
  );
}

export default Urls;
