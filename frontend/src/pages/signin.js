import * as React from 'react';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import FormControl from "@mui/material/FormControl";
import Checkbox from '@mui/material/Checkbox';
import Link from '@mui/material/Link';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Swal from 'sweetalert2';
import { withRouter } from "react-router-dom";
import '../style.css'
const theme = createTheme();


export default function SignIn() {
  let [link, setLink] = React.useState('');
  let [location, setLocation] = React.useState('');
  const [menu, setMenu] = React.useState('');

  let [loading, setLoading] = React.useState(false);
  const navigate = useNavigate();
  const validate = async (response) => {
    if (response["detail"] == "cold start" && menu == "1") {
      //cold start 시에 실행시켜야 하는 항목
      await Swal.fire({
        title: '싫어하는 음식을 선택해주세요.',
        html: `
        <input type="checkbox" id="c1"  /><label for="c1"></label>
        <input type="checkbox" id="c2"  /><label for="c2"></label>
        <input type="checkbox" id="c3"  /><label for="c3"></label><br>
        <input type="checkbox" id="c4"  /><label for="c4"></label>
        <input type="checkbox" id="c5"  /><label for="c5"></label>
        <input type="checkbox" id="c6"  /><label for="c6"></label><br>
        <input type="checkbox" id="c7"  /><label for="c7"></label>
        <input type="checkbox" id="c8"  /><label for="c8"></label>
        <input type="checkbox" id="c9"  /><label for="c9"></label>
        `,
        confirmButtonText: 'confirm',
        preConfirm: () => {
          var c1 = Swal.getPopup().querySelector('#c1').checked
          var c2 = Swal.getPopup().querySelector('#c2').checked
          var c3 = Swal.getPopup().querySelector('#c3').checked
          var c4 = Swal.getPopup().querySelector('#c4').checked
          var c5 = Swal.getPopup().querySelector('#c5').checked
          var c6 = Swal.getPopup().querySelector('#c6').checked
          var c7 = Swal.getPopup().querySelector('#c7').checked
          var c8 = Swal.getPopup().querySelector('#c8').checked
          var c9 = Swal.getPopup().querySelector('#c9').checked
          var count = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c9 + c9
          if (count <= 3) {
            return { c1: c1, c2: c2, c3: c3, c4: c4, c5: c5, c6: c6, c7: c7, c8: c8, c9: c9 }
          }
          else {
            Swal.showValidationMessage(
              '<i class="fa fa-info-circle"></i> 3개 까지만 선택해주세요'
            )
          }
        }
      }
      ).then((result) => {
        cold_start({
          name: link.substring(29, 53),
          location: location,
          menu: menu,
          c1: result.value.c1,
          c2: result.value.c2,
          c3: result.value.c3,
          c4: result.value.c4,
          c5: result.value.c5,
          c6: result.value.c6,
          c7: result.value.c7,
          c8: result.value.c8,
          c9: result.value.c9,
        });
      })

    }
    else if (response["detail"] == "cold start" && menu == "2") {
      cold_start({
        name: link.substring(29, 53),
        location: location,
        menu: menu,
        c1: false,
        c2: false,
        c3: false,
        c4: false,
        c5: false,
        c6: false,
        c7: false,
        c8: false,
        c9: false,
      });
    }
    else if (response["detail"] == "not cold start") {
      // 콜드스타트 아니고 기존에 실행히시켜야 하던 항목
      window.localStorage.setItem('restaurants0', JSON.stringify(response["restaurants0"]));
      window.localStorage.setItem('restaurants1', JSON.stringify(response["restaurants1"]));
      window.localStorage.setItem('restaurants2', JSON.stringify(response["restaurants2"]));
      window.localStorage.setItem('name', JSON.stringify(response["name"]));

      console.log(response);
      window.location = '/album'
    }
    if (response["detail"] == "low data") {
      Swal.fire({
        icon: 'error',
        title: 'Oops...',
        text: '인근에 음식점이 충분하지 않습니다',
      })
    }
    else {
      let timeoutId;
      Swal.fire({
        title: '로딩 중입니다',
        text: '잠시만 기다려주세요',
        icon: 'info',
        timer: 565555555,
        showConfirmButton: false,
        showCancelButton: true,
        cancelButtonText: 'Cancel'
      })
    }
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
    fetch('/api/signin', requestOptions)
      .then((response) => response.json())
      .then((response) => {
        validate(response);

      })
      .catch(error => alert(error.message));



  };
  const cold_start = async (userData) => {
    const requestOptions = {
      method: "POST",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    };
    await fetch('/api/signin/cold', requestOptions)
      .then((response) => response.json())
      .then((response) => {
        validate(response);

      })
      .catch(error => alert(error.message));



  };

  const howtouse = (event) => {
    Swal.fire({
      width: 800,
      html: `
      <div>
        <h1>머글끼니 사용 방법!</h1>
        <h2>1.
        <a href="https://naver.com">네이버 </a>로그인 후</h2>
        <h2>2.
        <a href="https://m.place.naver.com/my/">네이버 마이플레이스</a>
        - 클릭
        </h2>
        <h2>3.주소 복사</h2>
        <img src= 'img/howto1.jpg' width ='600'/>
        <h3>4.현재 서비스는 서울에 있는 식당을 기준으로 서비스 되고 있습니다. </h3>
        <h3>서울에서 추천을 원하는 주소를 입력해주세요.</h3>
      </div>
      `
    })
  }




  const handleChange = (event: SelectChangeEvent) => {
    setMenu(event.target.value);
  };

  const handleClick1 = (event) => {

    if (link.includes('place.naver.com/my')) {

      window.localStorage.setItem('link', link);
      signin({
        name: link.substring(29, 53),
        location: location,
        menu: menu,
      });
    }
    else {
      Swal.fire({
        title: '네이버 링크 오류',
        text: "하단 '어떻게 사용하나요' 참고!",
        icon: 'warning',

      })
    };
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
            머글끼니
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
              label="추천을 원하는 장소 ex) 강남역, 신촌역 등"
              name="location"
              onChange={(e) => {
                setLocation(e.target.value);
              }}
            />
            <FormControl fullWidth>
              <InputLabel id="demo-simple-select-label">식사 or 카페를 선택해주세요</InputLabel>
              <Select
                labelId="select1"
                id="select_id"
                value={menu}
                label="munu"
                onChange={handleChange}
              >
                <MenuItem value={1}>식사</MenuItem>
                <MenuItem value={2}>카페&디저트</MenuItem>
              </Select>
            </FormControl>

            <div>
              <Button
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
                <Link variant="body2" onClick={howtouse}>
                  어떻게 사용하나요?
                </Link>
              </Grid>
            </Grid>
          </Box>
        </Box>

      </Container>
    </ThemeProvider>
  );
}
