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
  console.log(JSON.parse(window.localStorage.getItem("name")))
  console.log(restaurants[0].length)
  const positive = (userData) => {
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
        console.log(response)
      })
      .catch(error => alert(error.message));
  };
  const negative = (userData) => {
    const requestOptions = {
      method: "POST",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    };
    fetch('/api/album/negative', requestOptions)
      .then((response) => response.json())
      .catch(error => alert(error.message));
  };
  const handleClick1 = (event) => {
    navigate('/signin');
  };

  const handleClick2 = (event) => {
    var user = window.localStorage.getItem('link').substring(29, 53)

    negative({
      user_id1: user,
      rest_id1: restaurants[0][card_num[0]]["id"],
      is_positive1: false,
      model1: restaurants[0][card_num[0]]["model"],
      user_id2: user,
      rest_id2: restaurants[1][card_num[1]]["id"],
      is_positive2: false,
      model2: restaurants[1][card_num[1]]["model"],
      user_id3: user,
      rest_id3: restaurants[2][card_num[2]]["id"],
      is_positive3: false,
      model3: restaurants[2][card_num[2]]["model"]
    });
    setCardNum([(card_num[0] + 1) % restaurants[0].length, (card_num[1] + 1) % restaurants[1].length, (card_num[2] + 1) % restaurants[2].length])


  };



  const handleClick4 = (event) => {
    // 일단 유저에서 열리게 
    var user = window.localStorage.getItem('link').substring(29, 53)
    const card = event.target.id
    const i = restaurants[card][card_num[card]]["id"]
    const url = "https://m.place.naver.com/restaurant/" + i + "/home"
    window.open(url, "_blank", "noopener, noreferrer");
    positive({
      user_id: user,
      rest_id: restaurants[card][card_num[card]]["id"],
      is_positive: true,
      model: restaurants[card][card_num[card]]["model"]
    });
  };

  const handleClick5 = (event) => {
    // 결과 리셋하는 코드 쓱쓱
    var user = window.localStorage.getItem('link').substring(29, 53)
    const card = event.target.id

    album({
      user_id: user,
      rest_id: restaurants[card][card_num[card]]["id"],
      is_positive: false,
      model: restaurants[card][card_num[card]]["model"]
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
            <Typography variant="h4" color="inherit" noWrap sx={{ marginBottom: 2 }}>
              {window.localStorage.getItem('name')}님의 추천 목록         </Typography>

            <Grid container spacing={3}>
              {/* cards의 card 가 하나씩 들어가는 반복문 */}
              {cards.map((card) => (
                <Grid item key={card} xs={12} sm={6} md={12}>
                  <Card
                    sx={{
                      height: '100%',
                      display: 'flex',
                    }}
                  >

                    <CardMedia
                      component="img"
                      md={{
                        // 16:9
                        pt: '100%',
                      }}
                      sx={{ position: "relative", objectFit: "contain", height: 250, left: -125 }}
                      image={restaurants[card][card_num[card]]["img_url"]}
                      alt="random"
                    />
                    <CardContent sx={{ position: "relative", objectFit: "contain", left: -100, width: 700 }}>
                      <Typography gutterBottom variant="h5" component="h2">
                        {restaurants[card][card_num[card]]["name"]}
                      </Typography>
                      <Typography>
                        {restaurants[card][card_num[card]]["tag"]}
                      </Typography>
                      <Typography sx={{ position: "relative", top: 10, fontSize: 18, width: 400 }}>
                        {window.localStorage.getItem('name') + restaurants[card][card_num[card]]["ment"]}
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button id={card} type="submit" size="small" onClick={handleClick4} sx={{ position: "relative", bottom: -90, width: 100, left: -130 }}>식당 링크 열기</Button>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
            <Stack
              sx={{ pt: 4 }}
              direction="row"
              spacing={2}
              justifyContent="none"
            >

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
            <Button variant="outlined" onClick={handleClick1} sx={{ position: "relative", top: -200, height: 50, right: 300 }}>
              처음 화면으로</Button>
            <Button variant="outlined" onClick={handleClick2} sx={{ position: "relative", top: -200, height: 50, left: 300 }}>
              다른 결과 보기</Button>
          </Box>
        </Container>


      </main>

    </ThemeProvider >
  );
}
