import React, { Component } from "react";
import * as d3 from "d3";
import ReactECharts from "echarts-for-react";
import axios from "axios";

import {
  RedoOutlined,
  CloseOutlined,
  HeatMapOutlined,
  AimOutlined,
  MenuOutlined,
  AppstoreOutlined,
  RadarChartOutlined,
} from "@ant-design/icons";
import {
  Button,
  Modal,
  Row,
  Col,
  Typography,
  Radio,
  Empty,
  Select,
  message,
  Tooltip,
  Tag,
  Image,
  Descriptions,
  Card,
  Spin,
} from "antd";
import ShowImg from "./ShowImg";

const { Option } = Select;
const { Title, Text } = Typography;
const xScale = d3.scaleLinear();
const yScale = d3.scaleLinear();
const maxValue = 40; 
const lineWidth = 0.1; 
const rows = 65; 
const cols = 70; 
const imgSize = 12; 

const mapColors = [
  "#e7f1ff",
  "#b3d2ed",
  "#5ca6d4",
  "#1970ba",
  "#0c3b80",
  "#042950",
];
const group_color = [
  "#6fe214",
  "#2e2b7c",
  "#c94578",
  "#ebe62b",
  "#f69540",
  "#9f9a9a", // back
];
const mapLevel = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"];
const category_name = ["no cancer", "cancer", "high cancer"];
const category_color = ["#40b373", "#d4b446", "#ee8826"]; // and more
const noise_tag_color = ["#90b7b7", "#a9906c", "#ef9a9a"];
const noise_tag_name = ["easy", "normal", "hard"];

export default class MapVision extends Component {
  constructor(props) {
    super(props);
    this.state = {
      heatEvent: null,
      heatMapType: "close",
      noiseChildren: [],
      index: [],
      choosePatches: props.choosePatches,

      load: false,
      visible: false,
      gridSize: 1,
      confirmLoading: false,
      submitLoading: false,
      barOption: {},
      areaOption: {},
      selectedPatch: 0,
      tipShowColor: [],
    };
  }
  noiseFilter = (value) => {
    var x_y = this.props.sample_Data.file_name[value].split("_");
    var y = parseInt(x_y[1]);
    var x = parseInt(x_y[3].split("png")[0]);
    this.setIndex(x, y);

    this.setVisible(true, value);
  };
  closeMap = () => {
    this.props.closeMap();
  };
  setIndex = (x, y) => {
    this.setState({
      index: [x, y],
    });
  };
  setVisible = async (args, key) => {
    var iter = [];
    for (var i = 1; i <= this.props.current_iteration; i++) iter.push(i);
    var o2us = this.props.sample_Data.o2us_num[key].slice(1, -1).split(",");
    var fines = this.props.sample_Data.fines_num[key].slice(1, -1).split(",");
    o2us.forEach(function (obj) {
      obj = parseFloat(obj);
    });
    fines.forEach(function (obj) {
      obj = parseFloat(obj);
    });
    await this.setState({
      barOption: {
        grid: {
          left: "30%",
          right: "10%",
        },
    
        yAxis: {
          type: "category",
          data: ["Fitting", "Alinment"],
        },
        xAxis: {
          type: "value",
          min: 0.0,
          max: 1.0,
          show: false,
          splitLine: {
            show: false,
          },
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        series: [
          {
            data: [
              {
                value: this.props.sample_Data.o2u[key],
                itemStyle: {
                  color: "#5470c6",
                },
              },
              {
                value: this.props.sample_Data.fine[key],
                itemStyle: {
                  color: "#93cd77",
                },
              },
            ],
            type: "bar",
          },
        ],
      },
      areaOption: {
        grid: {
          top: 0,
          left: "13%",
          right: "13%",
          bottom: 20,
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "cross",
            animation: false,
            label: {
              backgroundColor: "#505765",
            },
          },
        },
        xAxis: [
          {
            type: "category",
            boundaryGap: false,
            axisLine: { onZero: false },
            // prettier-ignore
            data: iter,
          },
        ],
        yAxis: [
          {
            name: "Fitting",
            type: "value",
            min: 0,
            max: 1,
          },
          {
            name: "Alignment",
            nameLocation: "start",
            alignTicks: true,
            type: "value",
            inverse: true,
            min: 0,
            max: 1,
          },
        ],
        series: [
          {
            name: "Fitting",
            type: "line",
            areaStyle: {},
            lineStyle: {
              width: 1,
            },
            showSymbol: false,
            emphasis: {
              focus: "series",
            },
            markArea: {
              silent: true,
              itemStyle: {
                opacity: 0.3,
              },
            },
            data: o2us,
          },
          {
            name: "Alignment",
            type: "line",
            yAxisIndex: 1,
            showSymbol: false,
            areaStyle: {},
            lineStyle: {
              width: 1,
            },
            emphasis: {
              focus: "series",
            },
            markArea: {
              silent: true,
              itemStyle: {
                opacity: 0.3,
              },
            },
            data: fines,
          },
        ],
      },
    });
    this.setState({
      visible: args,
    });
  };

  changeGridSize = async (e) => {
    this.setState({
      gridSize: e.target.value,
    });
    this.drawChart();
  };

  setConfirmLoading = (args) => {
    this.setState({
      confirmLoading: args,
    });
  };
  handleOk = () => {
  
    this.setConfirmLoading(true);
    setTimeout(() => {
      this.setVisible(false, 0);
      this.setConfirmLoading(false);
    }, 2000);
  };

  handleCancel = () => {
    this.setState({
      visible: false,
    });
  };

  submitForTrain = () => {
    var This = this;
    this.setState({
      submitLoading: true,
    });

    message.loading(
      "Next AL-Iteration is running in progress. Please wait for minutes...",
      () => {
        axios
          .post("http://127.0.0.1:5000/train", {
            iteration: This.props.current_iteration,
          })
          .then(function (response) {
            // update
            This.setState({
              submitLoading: false,
            });
            message.success("New dataset of model has been updated!", 2.5);
          });
      }
    );
  };
  componentDidMount = async () => {
    await this.props.onChildEvent(this);
    this.drawChart();
  };
  changeHeat = (e) => {
    this.setState({
      heatMapType: e.target.value,
    });
    var c = [];
    if (e.target.value == "kmeans_label") {
      for (var i = 0; i < group_color.length; i++) {
        c.push([
          i != group_color.length - 1 ? "group-" + i : "background",
          group_color[i],
        ]);
      }
    } else if (e.target.value == "category") {
      for (i = 0; i < category_color.length; i++) {
        c.push([category_name[i], category_color[i]]);
      }
    } else if (
      e.target.value == "o2u" ||
      e.target.value == "fine" ||
      e.target.value == "close"
    ) {
      for (i = 0; i < mapColors.length; i++) {
        c.push([mapLevel[i], mapColors[i]]);
      }
    }
    this.setState({
      tipShowColor: c,
    });
  };
  changeChoosePatches = async (p) => {
    await this.setState({
      choosePatches: p,
    });
  };
  deleteImg = async (index) => {
    var tags = this.state.choosePatches;
    tags = tags.filter((item) => item != index);

    await this.setState({
      choosePatches: tags,
    });
    this.props.changeDeletePatches(tags);
  };
  chooseImg = async (img_id) => {
    this.props.showMap(img_id);
  };
  setLoad = async (l) => {
    this.setState({
      load: l,
    });
  };

  drawChart = async () => {
    const colors = ["#e7f1ff", "#0c3b80"];
    const imgId = this.props.chooseMapImg;
    const This = this;
    xScale.domain([0, rows]).range([0, imgSize * rows]);
    yScale.domain([0, cols]).range([imgSize * cols, 0]);
    d3.select("#map").selectAll("svg").remove();
    const zoom = d3
      .zoom()
      .scaleExtent([1, maxValue])
      .translateExtent([
        [0, 0],
        [imgSize * rows, imgSize * cols],
      ])
      .on("zoom", zoomed);
    var colorScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range(colors)
      .interpolate(d3.interpolateHcl);
    const mainGroup = d3
      .select("#map")
      .append("svg")
      .attr("preserveAspectRatio", "xMinYMin meet")
      .attr("width", "100%")
      .attr("height", "100%");

    var imgData = [];
    var bk_imgData = [];
    var noiseChild = [];
    var backLocation = new Array(rows)
      .fill(0)
      .map((v) => new Array(cols).fill(0));
    var backLocationIndex = [];
    var data_len = Object.keys(this.props.sample_Data.class).length;
    var bk_data_len = Object.keys(this.props.bk_data.class).length;

    for (var i = 0; i < data_len; i++) {
      if (this.props.sample_Data.img_id[i] == imgId) {
        var x_y = this.props.sample_Data.file_name[i].split("_");
        var y = parseInt(x_y[1]);
        var x = parseInt(x_y[3].split("png")[0]);
        backLocation[y][x] = 1;
        imgData.push({
          patch_id: this.props.sample_Data.patch_id[i],
          o2u: this.props.sample_Data.o2u[i],
          grade: this.props.sample_Data.grade[i],
          fine: this.props.sample_Data.fine[i],
          heat_score: this.props.sample_Data.heat_score[i],
          file_name: this.props.sample_Data.file_name[i],
          noise: this.props.sample_Data.noise[i],
          class: this.props.sample_Data.class[i],
          //new
          o2us_num: this.props.sample_Data.o2us_num[i],
          fines_num: this.props.sample_Data.fines_num[i],
          CAM_file: this.props.sample_Data.CAM_file_name[i],
          is_labeled: this.props.sample_Data.is_labeled[i],
          kmeans_label: this.props.sample_Data.kmeans_label[i],
        });
        if (this.props.sample_Data.noise[i] > 0)
          noiseChild.push(
            <Option key={i}>{this.props.sample_Data.file_name[i]}</Option>
          );
      }
    }
    for (i = 0; i < bk_data_len; i++) {
      if (this.props.bk_data.img_id[i] == imgId) {
        var x_y = this.props.bk_data.file_name[i].split("_");
        var y = parseInt(x_y[1]);
        var x = parseInt(x_y[3].split("png")[0]);
        backLocation[y][x] = 1;
        bk_imgData.push({
          patch_id: this.props.bk_data.patch_id[i],
          file_name: this.props.bk_data.file_name[i],
          class: this.props.bk_data.class[i],
          kmeans_label: this.props.bk_data.kmeans_label[i],
        });
      }
    }
    var o2u_range = [2, -1];
    var fine_range = [2, -1];
    var smooth = 1e-5;
    for (var i = 0; i < imgData.length; i++) {
      o2u_range[0] =
        imgData[i]["o2u"] < o2u_range[0] ? imgData[i]["o2u"] : o2u_range[0];
      o2u_range[1] =
        imgData[i]["o2u"] > o2u_range[1] ? imgData[i]["o2u"] : o2u_range[1];
      fine_range[0] =
        imgData[i]["o2u"] < fine_range[0] ? imgData[i]["fine"] : fine_range[0];
      fine_range[1] =
        imgData[i]["fine"] > fine_range[1] ? imgData[i]["fine"] : fine_range[1];
    }
    for (var i = 0; i < imgData.length; i++) {
      imgData[i]["o2u"] =
        (imgData[i]["o2u"] - o2u_range[0]) /
        (o2u_range[1] - o2u_range[0] + smooth);
      imgData[i]["fine"] =
        (imgData[i]["fine"] - fine_range[0]) /
        (fine_range[1] - fine_range[0] + smooth);
    }
    var min_y = 100;
    var max_y = 0;
    for (var i = 0; i < rows; i++)
      for (var j = 0; j < cols; j++) {
        if (backLocation[i][j] == 1) {
          min_y = Math.min(i, min_y);
          max_y = Math.max(i, max_y);
        }
      }

    for (i = min_y; i <= max_y; i++) {
      var min_x = 100;
      var max_x = 0;
      for (j = 0; j < cols; j++)
        if (backLocation[i][j] == 1) {
          min_x = Math.min(j, min_x);
          max_x = Math.max(j, max_x);
        }
      for (j = min_x; j <= max_x; j++) {
        if (backLocation[i][j] == 0) backLocationIndex.push([i, j]);
      }
    }

    this.setState({
      noiseChildren: noiseChild,
    });

    mainGroup.call(zoom);
    drawGrid();
    drawPatches();

    d3.select("#zoom_out").on("click", () => {
      mainGroup.transition().call(zoom.transform, d3.zoomIdentity, [0, 0]);
      mainGroup.selectAll("rect").remove();
      This.setState({
        heatMapType: "close",
      });
    });

    d3.select("#zoom_change").on("change", toHeatMap);
    d3.select("#zoom_change1").on("change", toHeatMap);

 
    async function drawGrid(event) {
      mainGroup.selectAll("line").remove();
      var margin = lineWidth;
      if (event != null && event.transform.k > 25)
        margin = (lineWidth * event.transform.k) / 20;
      var grid = (g) =>
        g
          .attr("stroke", "blue")
          .attr("stroke-opacity", 0.5)
          .attr("stroke-width", margin)
          .call((g) =>
            g
              .append("g")
              .selectAll("line")
              .data(xScale.ticks(rows))
              .join("line")
              .attr("x1", (d) => xScale(This.state.gridSize * d))
              .attr("x2", (d) => xScale(This.state.gridSize * d))
              .attr("y2", imgSize * cols)
              .attr("transform", event == null ? null : event.transform)
          )
          .call((g) =>
            g
              .append("g")
              .selectAll("line")
              .data(yScale.ticks(cols))
              .join("line")
              .attr("y1", (d) => yScale(This.state.gridSize * d))
              .attr("y2", (d) => yScale(This.state.gridSize * d))
              .attr("x2", imgSize * rows)
              .attr("transform", event == null ? null : event.transform)
          );
      mainGroup.call(grid);
    }

    async function drawPatches(event) {
      mainGroup.selectAll("image").remove();

      const imgs = mainGroup.selectAll("image").data([0]);
      var margin = lineWidth;
      var p = process.env.REACT_APP_IMAGE_PATH_60;

      if (event != null) {
        margin = (lineWidth * event.transform.k) / 20;
        if (event.transform.k > 8) p = process.env.REACT_APP_IMAGE_PATH_224;
      }
      var transform_x =
        event == null
          ? 0
          : parseInt(-event.transform.x / (imgSize * event.transform.k));
      var transform_y =
        event == null
          ? 0
          : parseInt(-event.transform.y / (imgSize * event.transform.k));
      var transform_row =
        event == null ? rows : parseInt(rows / event.transform.k) + 1;
      var transform_col =
        event == null ? cols : parseInt(cols / event.transform.k) + 1;
      await imgData.forEach((img, key) => {
        var x_y = img["file_name"].split("_");
        var y = parseInt(x_y[1]);
        var x = parseInt(x_y[3].split("png")[0]);
        if (
          x >= transform_x &&
          x <= transform_x + transform_row &&
          y >= transform_y &&
          y <= transform_y + transform_col
        ) {
          var path = p + "image_" + imgId + "/x_" + y + "_" + "y_" + x + ".png";
          imgs
            .enter()
            .append("svg:image")
            .attr("xlink:href", path)
            .attr("row", x)
            .attr("col", y)
            .attr("x", imgSize * x + margin)
            .attr("y", imgSize * y + margin)
            .attr("img_id", imgId)
            .attr("patch_id", img["patch_id"])
            .attr("width", imgSize - lineWidth - margin)
            .on("mouseover", function (d) {
              d3.select(this)
                .attr("width", imgSize * 1.2)
                .attr("height", imgSize * 1.2);
            })
            .on("mouseout", function (d) {
              d3.select(this)
                .attr("width", imgSize - margin)
                .attr("height", imgSize - margin);
            })
            .on("click", async function (d) {
              await This.setIndex(
                this.getAttribute("row"),
                this.getAttribute("col")
              );
              await This.setState({
                selectedPatch: key,
              });
              This.setVisible(true, key);
            })
            .attr("transform", event == null ? null : event.transform);
        }
      });
      await bk_imgData.forEach((img, key) => {
        var x_y = img["file_name"].split("_");
        var y = parseInt(x_y[1]);
        var x = parseInt(x_y[3].split("png")[0]);
        if (
          x >= transform_x &&
          x <= transform_x + transform_row &&
          y >= transform_y &&
          y <= transform_y + transform_col
        ) {
          var path = p + "image_" + imgId + "/x_" + y + "_" + "y_" + x + ".png";
          imgs
            .enter()
            .append("svg:image")
            .attr("xlink:href", path)
            .attr("row", x)
            .attr("col", y)
            .attr("x", imgSize * x + margin)
            .attr("y", imgSize * y + margin)
            .attr("img_id", imgId)
            .attr("patch_id", img["patch_id"])
            .attr("width", imgSize - lineWidth - margin)
            .on("mouseover", function (d) {})
            .on("mouseout", function (d) {
              d3.select(this)
                .attr("width", imgSize - margin)
                .attr("height", imgSize - margin);
            })
            .attr("transform", event == null ? null : event.transform);
        }
      });
    }

    async function drawHeatMap() {
      if (This.state.heatMapType != "close") {
        mainGroup.selectAll("rect").remove();
        try {
          var margin = lineWidth / This.state.heatEvent.transform.k;
        } catch (err) {
          var margin = lineWidth;
        }
        setTimeout(() => {
          mainGroup.selectAll("rect").remove();

          imgData.forEach((img) => {
            var x_y = img["file_name"].split("_");
            var y = parseInt(x_y[1]);
            var x = parseInt(x_y[3].split("png")[0]);
            var lb = parseInt(img["class"]);
            mainGroup
              .append("g")
              .append("rect") 
              .attr("x", imgSize * x + margin)
              .attr("y", imgSize * y + margin)
              .attr("width", imgSize - margin)
              .attr("height", imgSize - margin)
              .attr("fill", function () {
                if (This.state.heatMapType == "category")
                  return category_color[lb - 1];
                else if (This.state.heatMapType == "kmeans_label")
                  return group_color[parseInt(img[This.state.heatMapType])];
                else if (This.state.heatMapType == "o2u")
                  return colorScale(
                    parseFloat(img[This.state.heatMapType] + 0.2)
                  );
                else return colorScale(parseFloat(img[This.state.heatMapType]));
              })
              .attr("opacity", 0.8)
              .attr(
                "transform",
                This.state.heatEvent == null
                  ? null
                  : This.state.heatEvent.transform
              );
          });
          bk_imgData.forEach((img) => {
            var x_y = img["file_name"].split("_");
            var y = parseInt(x_y[1]);
            var x = parseInt(x_y[3].split("png")[0]);
            var lb = parseInt(img["class"]);
            mainGroup
              .append("g")
              .append("rect") //添加类型
              .attr("x", imgSize * x + margin)
              .attr("y", imgSize * y + margin)
              .attr("width", imgSize - margin)
              .attr("height", imgSize - margin)
              .attr("fill", function () {
                if (This.state.heatMapType == "category")
                  return category_color[lb - 1];
                else if (This.state.heatMapType == "kmeans_label")
                  return group_color[parseInt(img[This.state.heatMapType])];
                else if (This.state.heatMapType == "kmeans_label")
                  return group_color[parseInt(img[This.state.heatMapType])];
                else if (This.state.heatMapType == "o2u")
                  return colorScale(0.1);
                else return colorScale(0.8);
              })
              .attr("opacity", 0.8)
              .attr(
                "transform",
                This.state.heatEvent == null
                  ? null
                  : This.state.heatEvent.transform
              );
          });
          backLocationIndex.forEach((item) => {
            mainGroup
              .append("g")
              .append("rect") 
              .attr("x", imgSize * item[1] + margin)
              .attr("y", imgSize * item[0] + margin)
              .attr("width", imgSize - margin)
              .attr("height", imgSize - margin)
              .attr("fill", function () {
                if (This.state.heatMapType == "category")
                  return category_color[0];
                else if (This.state.heatMapType == "o2u")
                  return colorScale(0.1);
                else return colorScale(0.8);
              })
              .attr("opacity", 0.8)
              .attr(
                "transform",
                This.state.heatEvent == null
                  ? null
                  : This.state.heatEvent.transform
              );
          });
        }, 0);
      }
    }

    async function toHeatMap() {
      var type = This.state.heatMapType;

      if (type != "close") {
        await drawHeatMap();
      } else {
        mainGroup.selectAll("rect").remove();
      }
    }
    async function zoomed(event) {
      This.setState({ heatEvent: event });
      await drawGrid(event);
      await drawPatches(event);
      await drawHeatMap(event);
    }
  };

  render() {
    return (
      <>
        <Card bordered={false} hoverable={true}>
          <Row align="center" gutter={[5, 3]}>
            <Col span={24}>
              <Row gutter={10}>
                <Col span={3}>
                  <Title level={5}>WSI-{this.props.chooseMapImg}</Title>
                </Col>
                <Col span={3}>
                  <Text type="secondary">
                    <AimOutlined />
                    &nbsp;Refresh:
                  </Text>
                </Col>
                <Col span={4}>
                  <Tooltip placement="top" title={"Click for initial position"}>
                    <Button
                      id="zoom_out"
                      type="primary"
                      shape="round"
                      size="small"
                      icon={<RedoOutlined />}
                      ghost
                    >
                      Refresh
                    </Button>
                  </Tooltip>
                </Col>
                <Col span={3}>
                  <Text type="secondary">
                    <AppstoreOutlined />
                    &nbsp;Grid Size:
                  </Text>
                </Col>
                <Col span={8}>
                  <Radio.Group
                    onChange={this.changeGridSize}
                    value={this.state.gridSize}
                  >
                    <Radio value={1}>1</Radio>
                    <Radio value={2}>2</Radio>
                    <Radio value={3}>3</Radio>
                    <Radio value={4}>4</Radio>
                  </Radio.Group>
                </Col>
                <Col offset={3} span={3}>
                  <Text type="secondary">
                    <MenuOutlined />
                    &nbsp;Noise List:
                  </Text>
                </Col>
                <Col span={4}>
                  <Select
                    allowClear
                    size="small"
   
                    placeholder="Check Noise"
                    onChange={this.noiseFilter}
                  >
                    {this.state.noiseChildren}
                  </Select>
                </Col>
                <Col span={2}>
                  <Text type="secondary">
                    <RadarChartOutlined />
                    Map:
                  </Text>
                </Col>
                <Col span={5}>
                  <Radio.Group
                    id="zoom_change"
                    value={this.state.heatMapType}
                    onChange={this.changeHeat}
                    size="small"
                    buttonStyle="solid"
                  >
                    <Radio.Button value="kmeans_label" selected>
                      Group
                    </Radio.Button>
                    <Radio.Button value="category">Category</Radio.Button>
                  </Radio.Group>
                </Col>
                <Col span={2}>
                  <Text type="secondary">
                    <HeatMapOutlined />
                    Noise:
                  </Text>
                </Col>
                <Col span={5}>
                  <Radio.Group
                    id="zoom_change1"
                    value={this.state.heatMapType}
                    onChange={this.changeHeat}
                    size="small"
                    buttonStyle="solid"
                  >
                    <Radio.Button value="o2u" selected>
                      Fitting
                    </Radio.Button>
                    <Radio.Button value="fine">Alignment</Radio.Button>
                  </Radio.Group>
                </Col>
              </Row>
            </Col>
            <Col span={24}>
              <Spin id="loading" size="large" spinning={this.state.load}>
                <div id="map">
                  <div id="tooltipMap">
                    <Row gutter={5}>
                      {this.state.tipShowColor.map((item, index) => {
                        return (
                          <>
                            <Col offset={2} span={4}>
                              <div
                                style={{
                                  width: 15,
                                  height: 15,
                                  backgroundColor: item[1],
                                }}
                              ></div>
                            </Col>
                            <Col span={18}>
                              <Text>{item[0]}</Text>
                            </Col>
                          </>
                        );
                      })}
                    </Row>
                  </div>
                </div>
              </Spin>
            </Col>
            <Col span={24}>
              <Card style={{ height: 0 }}>
                {this.state.choosePatches.length === 0 ? (
                  <Empty />
                ) : (
                  this.state.choosePatches.map((item, index) => {
                    return (
                      <Card.Grid
                        className="childCard"
                        id={"childCard" + item}
                        key={"childCard" + item}
                      >
                        <Row align="top">
                          <Col span={22}>
                            <div className="WSI-tip">{item}</div>
                            <Image
                              className="grid-img"
                              preview={false}
                              src={
                                process.env.REACT_APP_WSI_PATH +
                                "/" +
                                item +
                                ".png"
                              }
                              onClick={() => this.chooseImg(item)}
                            />
                          </Col>
                          <Col span={2}>
                            <Button
                              danger
                              style={{ marginLeft: -6 }}
                              type="text"
                              icon={<CloseOutlined />}
                              size="small"
                              onClick={() => this.deleteImg(item)}
                            />
                            <div
                              style={{
                                height:
                                  8 * this.props.WSI_Data["grades2"][item] +
                                  "vh",
                                width: 10,
                                backgroundColor: noise_tag_color[2],
                              }}
                            ></div>
                            <div
                              style={{
                                height:
                                  8 * this.props.WSI_Data["grades1"][item] +
                                  "vh",
                                width: 10,
                                backgroundColor: noise_tag_color[1],
                              }}
                            ></div>
                            <div
                              style={{
                                height:
                                  8 * this.props.WSI_Data["grades0"][item] +
                                  "vh",
                                width: 10,
                                backgroundColor: noise_tag_color[0],
                              }}
                            ></div>
                          </Col>
                        </Row>
                      </Card.Grid>
                    );
                  })
                )}
              </Card>
            </Col>
          </Row>
        </Card>

        <Modal
          title={"Check Index:" + this.state.index}
          visible={this.state.visible}
          onOk={this.handleOk}
          confirmLoading={this.state.confirmLoading}
          onCancel={this.handleCancel}
        >
          <Row gutter={[0, 10]} justify="space-around" align="middle">
            <Col span={12} offset={1}>
              <ShowImg
                index={this.state.index}
                imgId={this.props.chooseMapImg}
              />
            </Col>
            <Col span={10} offset={1}>
              <Text type="secondary">Grad-Cam Image</Text>
              <Image
                // fallback="data:image/png;base64,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"
                width={160}
                height={160}
                src={
                  process.env.REACT_APP_IMAGE_CAM +
                  "/image_" +
                  this.props.chooseMapImg +
                  "/x_" +
                  this.state.index[0] +
                  "_y_" +
                  this.state.index[1] +
                  ".png"
                }
              />
            </Col>
            <Col span={24}>
              <Descriptions layout="vertical" size="large" column={4} bordered>
                <Descriptions.Item span={2} label="Fitting & Alignment Scores">
                  <ReactECharts
                    style={{ width: 190, height: 80 }}
                    option={this.state.barOption}
                  />
                </Descriptions.Item>
                <Descriptions.Item span={2} label="Trends">
                  <ReactECharts
                    style={{ width: 200, height: 110 }}
                    option={this.state.areaOption}
                  />
                </Descriptions.Item>
                <Descriptions.Item span={2} label="CC Grade">
                  <Tag
                    color={
                      noise_tag_color[
                        this.props.sample_Data.grade[this.state.selectedPatch]
                      ]
                    }
                  >
                    {
                      noise_tag_name[
                        this.props.sample_Data.grade[this.state.selectedPatch]
                      ]
                    }
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item span={2} label="Update Class">
                  <Select
                    style={{ width: "100%" }}
                    placeholder="select one class"
                    optionLabelProp="label"
                  >
                    {category_name.map((item, index) => {
                      return (
                        <Option value={index} label={item}>
                          <div className="demo-option-label-item">
                            <Tag color={category_color[index]}> {item}</Tag>
                          </div>
                        </Option>
                      );
                    })}
                  </Select>
                </Descriptions.Item>
              </Descriptions>
            </Col>
          </Row>
        </Modal>
      </>
    );
  }
}
