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


export default function HowUse() {
  const navigate = useNavigate();

  const handleClick = (event) => {
    navigate('/signin');
  };
  return (
    <Container component="main" maxWidth="md">
      <div>
        <h1>렉카메추 사용 방법!</h1>
        <h2>1.
        <a href="https://naver.com">네이버 </a>로그인 후</h2>
        <h2>2.
        <a href="https://m.place.naver.com/my/">네이버 마이플레이스</a>
        - 클릭
        </h2>
        <h2>3.주소 복사</h2>
        <img src= 'img/howto1.jpg' width ='800'/>
        <h3>4.현재 서비스는 서울에 있는 식당을 기준으로 서비스 되고 있습니다. 
          서울에서 추천을 원하는 주소를 입력해주세요.</h3>
      </div>
      <div>
      
      </div>
      
      <Grid container>
        <Grid item xs>
          {/* <Link variant="body2" onClick={handleClick}>돌아가기</Link> */}
          <Button variant="outlined" onClick={handleClick}>
                처음 화면으로
              </Button>
        </Grid>
        
      </Grid>
    </Container>
  );
}
