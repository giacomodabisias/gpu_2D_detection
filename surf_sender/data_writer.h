#ifndef ERROR_H
#define ERROR_H
#include <mutex>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/common/thread.hpp>
#include <sstream>
#include <fstream>


class DataWriter
{
public:
  typedef websocketpp::client<websocketpp::config::asio_client> client;
  typedef websocketpp::lib::lock_guard<websocketpp::lib::mutex> scoped_lock;

  const std::string uri_;
  client m_client_;
  websocketpp::connection_hdl m_hdl_;
  websocketpp::lib::mutex m_lock_;
  bool m_open_;
  bool m_done_;
  std::vector<std::string> * buffer_;
  client::connection_ptr con_;
  websocketpp::lib::error_code ec_;
  websocketpp::error::category cat_;

  DataWriter(const std::string & uri);

  void
  writeData(const std::string  s);

};

#endif