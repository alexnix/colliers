import React from "react";
import axios from "axios";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";
import "./App.css";

class App extends React.Component {
  constructor() {
    super();
    this.state = {
      data: null
    };
  }

  componentDidMount() {
    axios.get("http://localhost:3030/api").then(res => {
      this.setState({
        data: res.data
      });
    });
  }

  render() {
    const { data } = this.state;

    if (data != null) {
      const mean_predict = data["mean_predict"];
      const ground_truth = data["ground truth"];
      const X = data["X"];

      // const collegues = JSON.parse(data["collegues"]);
      const rows = [];
      for (const key in mean_predict) {
        const Xi = JSON.parse(X[key]);
        console.log(Xi);
        rows.push(
          <TableRow key={key}>
            <TableCell>{Xi["ID"]}</TableCell>
            <TableCell>
              Bucuresti {Xi["Nume Artera"]} {Xi["Numar"]}
            </TableCell>
            <TableCell>{Xi["Latitudine"]}</TableCell>
            <TableCell>{Xi["Longitudine"]}</TableCell>
            <TableCell>{Xi["An"]}</TableCell>
            <TableCell>{Xi["Suprafata construita desfasurata"]}</TableCell>
            <TableCell>{Xi["Suprafata teren"]}</TableCell>
            <TableCell>{Xi["Finisaje"]}</TableCell>
            <TableCell align="left" component="th" scope="row">
              {ground_truth[key]} €
            </TableCell>
            <TableCell align="right">{mean_predict[key]} €</TableCell>
            <TableCell align="right">
              {Number(mean_predict[key].replace(",", "")) -
                Number(ground_truth[key].replace(",", ""))}
            </TableCell>
          </TableRow>
        );
      }

      return (
        <div className="App">
          <h1>Predictii pe baza a 150 de comparabile (Bucuresti)</h1>
          <Paper className="table-paper-wrapper">
            <Table aria-label="simple table">
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Adresa</TableCell>
                  <TableCell>Lat</TableCell>
                  <TableCell>Lng</TableCell>
                  <TableCell>An</TableCell>
                  <TableCell>Supragata totala</TableCell>
                  <TableCell>Suprafata construita</TableCell>
                  <TableCell>Finisaje</TableCell>
                  <TableCell>ground_truth (PRET REAL)</TableCell>
                  <TableCell align="right">
                    Predict (Ansamble: SVM + RF + KNN)
                  </TableCell>
                  <TableCell align="right">Error</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>{rows}</TableBody>
            </Table>
          </Paper>
        </div>
      );
    }

    return (
      <div className="App">
        <p>Loading</p>
      </div>
    );
  }
}

export default App;
