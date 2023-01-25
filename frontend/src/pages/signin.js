import * as React from 'react';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Link from '@mui/material/Link';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';

import Swal from 'sweetalert2';
import { withRouter } from "react-router-dom";

//https://m.place.naver.com/my/5c667add298eafd0547442d8/feed

function Copyright(props) {
  return (
    <Typography
      variant="body2"
      color="text.secondary"
      align="center"
      {...props}
    >
      {'Copyright © '}
      <Link color="inherit" href="https://mui.com/">
        Your Website
      </Link>{' '}
      {new Date().getFullYear()}
      {'.'}
    </Typography>
  );
}







const theme = createTheme();


export default function SignIn() {
  let [link, setLink] = React.useState('');
  let [location, setLocation] = React.useState('');
  
  let [loading, setLoading] = React.useState(false);
  const navigate = useNavigate();
  const validate = (response) => {
    
    // if (response["state"] === "Approved") {
    //   console.log(response["restaurants"]);

    // }
    // else if (response["detail"] === "wrong password"){

    // }
    window.localStorage.setItem('restaurants1', JSON.stringify(response["restaurants1"]));
    window.localStorage.setItem('restaurants2', JSON.stringify(response["restaurants2"]));
    window.localStorage.setItem('restaurants3', JSON.stringify(response["restaurants3"]));

    console.log(response);
  };
  const signin = (userData) => {
    const requestOptions = {
      method: "POST",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    };
    fetch('http://localhost:8001/signin', requestOptions)
      .then((response) => response.json())
      .then((response) => {
        validate(response);
        
      })
      .catch(error => alert(error.message));

    

  };

  let timeoutId;

  const showAutoClose = () => {
    const loading = Swal.fire({
      title: 'Auto Close Alert',
      text: 'This alert will close in 5 seconds.',
      icon: 'info',
      timer: 5000,
      showConfirmButton : false,
      showCancelButton: true,
      cancelButtonText: 'Cancel',

      // onOpen: () => {
      //   Swal.showLoading()
      //   timeoutId = setTimeout(() => {
          
      //     loading.close();
          
      //   }, 5000);
      // },
      // onClose: () => {
      //   clearTimeout(timeoutId);
      //   console.log('Alert closed')
        
      // }

    }).then((result) => {
      if (result.dismiss === Swal.DismissReason.cancel) {
        clearTimeout(timeoutId);
        console.log('Cancelled');
        // window.location = '/signin'
      } else {
        window.location ='/album'
      }
    });
  }






  const handleClick1 = (event) => {
    
    if (link.includes('place.naver.com/my')) {
      
      window.localStorage.setItem('link', link);
      signin({
        name: link.substring(29,53),
        location: location,
      });


      showAutoClose()
      // navigate('/album');
      // window.location = '/album';
    }
    else {
      Swal.fire({
        title: '네이버 링크 오류',
        text: "하단 '어떻게 사용하나요' 참고!",
        icon: 'warning',

      })
    };
  };


  const handleClick2 = (event) => {
    //showLoading()
    navigate('/howuse');
  };


  return (
    <ThemeProvider theme={theme}>
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <Box
          sx={{
            marginTop: 8,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Typography component="h1" variant="h5">
            네이버 맛집 추천
          </Typography>
          <Box component="form" noValidate sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="link"
              label="네이버 My Place Link"
              name="link"
              autoFocus
              onChange={(e) => {
                setLink(e.target.value);
              }}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              id="location"
              label="추천을 원하는 주소"
              name="location"
              onChange={(e) => {
                setLocation(e.target.value);
              }}
            />

            <FormControlLabel
              control={<Checkbox value="remember" color="primary" />}
              label="Remember me"
            />
            <div>
              <Button
                // type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
                onClick={handleClick1}
              >
                추천
              </Button>
            </div>
            <Grid container>
              <Grid item xs>
                <Link variant="body2" onClick={handleClick2}>
                  어떻게 사용하나요?
                </Link>
              </Grid>
            </Grid>
          </Box>
        </Box>
        <Copyright sx={{ mt: 8, mb: 4 }} />
      </Container>
    </ThemeProvider>
  );
}
