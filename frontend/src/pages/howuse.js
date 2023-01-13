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
    <Container component="main" maxWidth="xs">
      How Use
      <Grid container>
        <Grid item xs>
          <Link variant="body2" onClick={handleClick}>돌아가기</Link>
        </Grid>
      </Grid>
    </Container>
  );
}
