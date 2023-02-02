import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import CssBaseline from '@mui/material/CssBaseline';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Link from '@mui/material/Link';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { redirect, renderMatches, useNavigate } from 'react-router-dom';




const cards = [0, 1, 2];

let initialCounters = [
  0, 0, 0, 0
];
const theme = createTheme();

export default function Album() {
  const navigate = useNavigate();
  const restaurants0 = JSON.parse(window.localStorage.getItem("restaurants0"));
  const restaurants1 = JSON.parse(window.localStorage.getItem("restaurants1"));
  const restaurants2 = JSON.parse(window.localStorage.getItem("restaurants2"));
  const restaurants = [restaurants0, restaurants1, restaurants2]
  let [card_num, setCardNum] = React.useState(initialCounters);

  const idx = [0, 0, 0, 0]

  console.log(restaurants)
  console.log(card_num)

  const album = (userData) => {
    const requestOptions = {
      method: "POST",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    };
    fetch('/api/album', requestOptions)
      .then((response) => response.json())
      .then((response) => {
        // validate(response);
        console.log(response)
      })
      .catch(error => alert(error.message));



  };

  const handleClick1 = (event) => {
    navigate('/signin');
  };

  const handleClick4 = (event) => {
    // 일단 유저에서 열리게 
    var user = window.localStorage.getItem('link').substring(29, 53)
    const card = event.target.id
    const i = restaurants[card][card_num[card]]["id"]
    const url = "https://m.place.naver.com/restaurant/" + i + "/home"
    window.open(url, "_blank", "noopener, noreferrer");
    // window.open(url, "_blank", "noopener, noreferrer");
    album({
      user_id: user,
      rest_id: restaurants[card][card_num[card]]["id"],
      is_positive: true
    });
  };

  const handleClick5 = (event) => {
    // 결과 리셋하는 코드 쓱쓱
    var user = window.localStorage.getItem('link').substring(29, 53)
    const card = event.target.id

    album({
      user_id: user,
      rest_id: restaurants[card][card_num[card]]["id"],
      is_positive: false
    });
    if (card == 0) {
      setCardNum([card_num[0] + 1, card_num[1], card_num[2], card_num[3]])
    }
    else if (card == 1) {
      setCardNum([card_num[0], card_num[1] + 1, card_num[2], card_num[3]])
    }
    else if (card == 2) {
      setCardNum([card_num[0], card_num[1], card_num[2] + 1])
    }
    console.log(card)
    console.log(typeof (card))
    console.log(card_num)
    console.log(idx)
  };

  return (
    <ThemeProvider theme={theme}>


      <main>
        {/* Hero unit */}
        <Box
          sx={{
            bgcolor: 'background.paper',
            pt: 8,
            pb: 6,
          }}
        >

          <Container maxWidth="md">
            <Typography variant="h4" color="inherit" noWrap>
              {window.localStorage.getItem('link').substring(29, 53)}님의 추천 목록          </Typography>

            <Grid container spacing={3}>
              {/* cards의 card 가 하나씩 들어가는 반복문 */}
              {cards.map((card) => (
                <Grid item key={card} xs={12} sm={6} md={4}>
                  <Card
                    sx={{
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >

                    <CardMedia
                      component="img"
                      md={{
                        // 16:9
                        pt: '100%',
                      }}
                      image={restaurants[card][card_num[card]]["img_url"]}
                      alt="random"
                    />
                    <CardContent sx={{ flexGrow: 1 }}>
                      <Typography gutterBottom variant="h5" component="h2">
                        {restaurants[card][card_num[card]]["name"]}
                      </Typography>
                      <Typography>
                        {restaurants[card][card_num[card]]["tag"]}
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button id={card} type="submit" size="small" onClick={handleClick4}>식당 링크 열기</Button>
                      <Button id={card} type="submit" size="small" onClick={handleClick5}>다른 결과 보기</Button>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
            <Stack
              sx={{ pt: 4 }}
              direction="row"
              spacing={2}
              justifyContent="center"
            >
              {/* <Button variant="contained">더보기</Button> */}

            </Stack>
          </Container>
        </Box>
        <Container sx={{ py: 8 }} maxWidth="md">
          {/* End hero unit */}


          <Box
            m={1}
            display="flex"
            justifyContent="center"
            alignItems="center"
            sx={{
              bgcolor: 'background.paper',
              pt: 8,
              pb: 6,
            }}
          >
            <Button variant="outlined" onClick={handleClick1}>
              처음 화면으로</Button>
          </Box>
        </Container>


      </main>

    </ThemeProvider>
  );
}
