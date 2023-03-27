/*
 * @Date: 2022-05-26 22:06:56
 * @LastEditors: JZY
 * @LastEditTime: 2023-02-05 21:58:16
 * @FilePath: /project/Visual/src/components/CoreModule/ScatterModel/Scatter.jsx
 */
import * as d3 from "d3";
import React, { Component } from "react";
import ReactECharts from "echarts-for-react";
import {
  Row,
  Col,
  Typography,
  Select,
  Radio,
  Slider,
  Menu,
  Dropdown,
  Button,
  Descriptions,
  Card,
  Image,
  Switch,
} from "antd";
import {
  FilterOutlined,
  UnorderedListOutlined,
  ColumnWidthOutlined,
  TagOutlined,
} from "@ant-design/icons";
import "./index.css";

// global variable
const { Option } = Select;
const { Text } = Typography;
const labelStyle = { fontWeight: 500 };

const category_name = ["no cancer", "cancer", "high cancer"];
const category_color = ["#40b373", "#d4b446", "#ee8826"]; // and more
const group_color = [
  "#6fe214",
  "#2e2b7c",
  "#c94578",
  "#ebe62b",
  "#f69540",
];
const noise_tag_color = ["#90b7b7", "#a9906c", "#ef9a9a"];
const noise_tag_name = ["easy", "normal", "hard"];
const diff_shape = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle];
const rangeX = [1000, 0]
const rangeY = [1000, 0]

const epoches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

export default class Scatter extends Component {
  constructor(props) {
    super(props);
    this.state = {
      //
      grade: -1, // 筛选等级
      category: -1, // 筛选类别
      ImageId: -1,
      showGroup: false,
      noise_matrix: "o2u", // 筛选噪声指标
      range: [0, 1], // 筛选噪声指标范围
      //x,y,imgid,patchid,category,value,grade0~2 number
      hover_data: [],
      //
      selectPatches: props.patches,
      option: {},
    };
  }
  componentDidMount() {
    this.props.onChildEvent(this);
    this.drawChart();
  }
  //
  selectEpoch = (value) => {
    // console.log("changed", value);
  };
  selectGrade = (index, e) => {
    this.setState({ grade: parseInt(index) });
    // console.log(parseInt(index));
    this.drawChart();
  };
  selectCategory = (index, e) => {
    this.setState({ category: parseInt(index) });
    this.drawChart();
  };
  selectNoiseMatrix = (index) => {
    this.setState({ noise_matrix: index.target.value });
    this.drawChart();
  };
  selectRange = (value) => {
    this.setState({ range: value });
    this.drawChart();
  };
  toGroup = (checked: boolean) => {
    this.setState({
      showGroup: checked,
    });
    this.drawChart();
  };
  changeImageId = (value) => {
    var pre = this.state.ImageId;
    this.setState({
      ImageId: pre == value ? -1 : value,
    });
  };
  drawChart = () => {
    d3.select("#scatter").selectAll("svg").remove();
    const width = document.getElementById("scatter").clientWidth;
    const height = document.getElementById("scatter").clientHeight;

    const margin = { top: 10, right: 8, bottom: 20, left: 30 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    var This = this; //
    const svg = d3
      .select("#scatter")
      .append("svg")
      .attr("preserveAspectRatio", "xMinYMin meet")
      .attr("width", "100%")
      .attr("height", "100%");
    const xScale = d3.scaleLinear();
    const yScale = d3.scaleLinear();


    const mainGroup = svg
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // var scatter = mainGroup.append("g").attr("class", "scatters");
    setTimeout(() => {
      drawCricle();
    }, 0);

    async function drawCricle(event) {
      var sample_data = [];
      // var data_len = Object.keys(This.props.sample_Data.class).length;
      var data_len = 10000;
      if (event != null) {
        mainGroup.selectAll("path").remove();
      }

      var grade = This.state.grade;
      var category = This.state.category;
      var noise_matrix = This.state.noise_matrix;
      var range = This.state.range;
      var ImageId = This.state.ImageId;
      var showGroup = This.state.showGroup;


      for (var i = 0; i < data_len; i++) {
        var score = This.props.sample_Data[noise_matrix][i];
        var grade_ = grade >= 0 ? This.props.sample_Data.grade[i] == grade : 1;
        var category_ =
          category >= 0 ? This.props.sample_Data.class[i] == category : 1;
        var ImageId_ =
          ImageId > 0 ? This.props.sample_Data.img_id[i] == ImageId : 1;

        if (
          ImageId_ &&
          grade_ &&
          category_ &&
          score > range[0] &&
          score < range[1]
        ) {
          rangeX[0] = Math.min(rangeX[0], This.props.sample_Data.scatter_x[i])
          rangeX[1] = Math.max(rangeX[1], This.props.sample_Data.scatter_x[i])
          rangeY[0] = Math.min(rangeY[0], This.props.sample_Data.scatter_y[i])
          rangeY[1] = Math.max(rangeY[1], This.props.sample_Data.scatter_y[i])
          sample_data.push([
            This.props.sample_Data.scatter_x[i],
            This.props.sample_Data.scatter_y[i],
            This.props.sample_Data.grade[i],
            This.props.sample_Data.patch_id[i],
            This.props.sample_Data.img_id[i],
            This.props.sample_Data[noise_matrix][i],
            This.props.sample_Data.class[i],
            // new
            This.props.sample_Data.grades_num[i],
            This.props.sample_Data.o2us_num[i],
            This.props.sample_Data.fines_num[i],
            This.props.sample_Data.file_name[i],
            This.props.sample_Data.kmeans_label[i],
          ]);
        }
      }
      // 圆点
      var scatterCmbo = mainGroup
        .selectAll("g")
        .data(sample_data)
        .enter();
      // .append("g");
      xScale.domain(rangeX).range([0, innerWidth]);
      yScale.domain(rangeY).range([innerHeight, 0]);
      scatterCmbo
        .append("path")
        .attr("class", "scatter")
        .attr(
          "d",
          d3
            .symbol()
            .type((d) => diff_shape[d[2]])
            .size(
              ImageId == -1
                ? event == null
                  ? 15
                  : 15 * event.transform.k
                : event == null
                  ? 40
                  : 40 * event.transform.k
            )
        )
        .attr("transform", function (d) {
          if (event == null)
            return "translate(" + xScale(d[0]) + "," + yScale(d[1]) + ")";
          else {
            var t = event.transform;
            console.log(t)
          }
        })
        // 属性
        .attr("hover_data", (d) => d)
        //
        .attr("fill", function (d) {
          if (showGroup) return group_color[d[11]];
          else return noise_tag_color[d[2]];
        })
        .attr("fill-opacity", (d) => d[5] + 0.2)
        .on("mouseover", function (d, i) {
          d3.select(this).classed("circle-hover", true);
          // var hd = this.getAttribute('hover_data').split(",")
          var data = i[7].slice(1, -1).split(".");
          This.setState({
            hover_data: i,
            
            option: {
              title:{
                text:'Grades records:',
                textStyle:{
                  fontSize:15
                }
              },
              grid: {
                left: "40%",
                right: 0,
                top: "20%",
              },
              yAxis: {
                type: "category",
                data: noise_tag_name,
              },
              xAxis: {
                type: "value",
                boundaryGap: [0, 0.1],
              },
              series: [
                {
                  data: [
                    parseInt(data[0]),
                    parseInt(data[1]),
                    parseInt(data[2]),
                  ],
                  type: "bar",
                },
              ],
            },
          });
          d3.select("#tooltipScatter")
            .style("left", d.layerX + 10 + "px")
            .style("top", d.layerY - 50 + "px");
          d3.select("#tooltipScatter").classed("hidden", false);
        })
        .on("mouseout", function () {
          d3.select(this).classed("circle-hover", false);
          d3.select("#tooltipScatter").classed("hidden", true);
        })
        .on("click", function (d, i) {
          // This.props.changeBarRange(this.getAttribute("hover_data").split(",")[4])
          This.props.changeBarRange(i[4]);
          This.changeImageId(i[4]);
          This.drawChart();
        });
    }
  };

  render() {
    return (
      <>
        <Row id="scatterRow">
          <Col span={24} id="scatter" className="scatterChart">
            <Row>
              <Col offset={18} span={4}>
                <Dropdown
                  // open={true}
                  // onOpenChange={{}}
                  // id="scatterTipBox"
                  trigger="click"
                  placement="bottom"
                  overlay={
                    <Menu id="scatterTip">
                      <Row gutter={[0, 8]}>
                        <Col span={24}>
                          <Text className="tooltipText" type="secondary">
                            <TagOutlined /> Epoch:
                          </Text>
                        </Col>
                        <Select
                          defaultValue={this.props.current_iteration}
                          onChange={this.selectEpoch}
                          style={{
                            width: 100,
                          }}
                          disabled
                        >
                          {epoches.map((item, index) => {
                            return <Option value={item}>{item}</Option>;
                          })}
                        </Select>

                        <Col span={24}>
                          <Text className="tooltipText" type="secondary">
                            <UnorderedListOutlined />
                            CC Grade
                          </Text>
                        </Col>
                        {noise_tag_color.map((item, index) => {
                          return (
                            <>
                              <Col span={4}>
                                <div
                                  style={{
                                    width: 15,
                                    height: 15,
                                    backgroundColor: item,
                                  }}
                                ></div>
                              </Col>
                              <Col span={20}>
                                <Text>{noise_tag_name[index]}</Text>
                              </Col>
                            </>
                          );
                        })}
                        <Col span={24}>
                          <Text className="tooltipText" type="secondary">
                            <FilterOutlined />
                            CC Grade:
                          </Text>
                        </Col>
                        <Col span={24}>
                          <Select defaultValue="-1" onChange={this.selectGrade}>
                            <Option value="-1">All Sample</Option>
                            <Option value="0">easy</Option>
                            <Option value="1">normal</Option>
                            <Option value="2">hard</Option>
                          </Select>
                        </Col>
                        <Col span={24}>
                          <Text className="tooltipText" type="secondary">
                            <FilterOutlined />
                            Category:
                          </Text>
                        </Col>
                        <Col span={24}>
                          <Select
                            defaultValue="-1"
                            onChange={this.selectCategory}
                          >
                            <Option value="-1">All Sample</Option>
                            {category_name.map((item, index) => {
                              return <Option value={index + 1}>{item}</Option>;
                            })}
                          </Select>
                          {/* <Radio.Group
                            onChange={this.selectCategory}
                            defaultValue="-1"
                          >
                            <Radio.Button value="-1">All</Radio.Button>
                            {
                              category_name.map((item, index) => {
                                return <Radio.Button value={index}>
                                {item}
                              </Radio.Button>
                              })
                            }
                            
                          </Radio.Group> */}
                        </Col>
                        <Col span={24}>
                          <Text className="tooltipText" type="secondary">
                            <FilterOutlined />
                            Noise Matrix:
                          </Text>
                        </Col>
                        <Col span={24}>
                          <Radio.Group
                            onChange={this.selectNoiseMatrix}
                            defaultValue="o2u"
                          >
                            <Radio.Button value="o2u">Fitting</Radio.Button>
                            <Radio.Button value="fine">Alignment</Radio.Button>
                          </Radio.Group>
                        </Col>
                        <Col span={12}>
                          <Text className="tooltipText" type="secondary">
                            <FilterOutlined />
                            Groups:
                          </Text>
                        </Col>
                        <Col offset={1} span={8}>
                          <Switch
                            defaultChecked={false}
                            onClick={this.toGroup}
                          />
                        </Col>
                        <Col span={24}>
                          <Text className="tooltipText" type="secondary">
                            <ColumnWidthOutlined />
                            Value Range:{this.state.range[0]}~
                            {this.state.range[1]}
                          </Text>
                        </Col>
                        <Col span={24}>
                          <Slider
                            range
                            min={0}
                            max={1}
                            defaultValue={[0, 1]}
                            step={0.01}
                            onChange={this.selectRange}
                          />
                        </Col>
                      </Row>
                    </Menu>
                  }
                >
                  <Button id="scatterTipBox">Scatter for Patches Tool</Button>
                </Dropdown>
              </Col>
            </Row>
          </Col>

          <Card
            id="tooltipScatter"
            className="hidden"
            style={{ height: "25vh" }}
          >
            <Row>
              <Col span={14}>
                <Card
                  style={{ width: 100, height: 100 }}
                  cover={
                    <img
                      alt="example"
                      src={
                        process.env.REACT_APP_IMAGE_PATH_224 + "/image_" +
                        this.state.hover_data[4] +
                        "/" +
                        this.state.hover_data[10]
                      }
                    />
                  }
                >

                </Card>
              </Col>
              <Col span={10}>
                <Row>
                  <Col span={24}><Text type="secondary">
                    WSI-{this.state.hover_data[4]}
                  </Text></Col>
                  <Col span={24}><Text type="secondary">
                  patch:{this.state.hover_data[3]}
                  </Text></Col>
                  <Col span={24}><Text type="secondary">
                  value:{parseFloat(this.state.hover_data[5]).toFixed(3)}
                  </Text></Col>
                  <Col span={24}><Text type="secondary">
                  label:{this.state.hover_data[6]}
                  </Text></Col>
                </Row>
            
              </Col>
            <Col>
            <ReactECharts
                    option={this.state.option}
                    style={{ height: "16vh", width: "10vw" }}
                  />
            </Col>
            </Row>
          </Card>
        </Row>
      </>
    );
  }
}
